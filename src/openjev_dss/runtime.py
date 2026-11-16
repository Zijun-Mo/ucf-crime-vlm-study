"""Reusable training runtime and frozen visual cache provider."""
import hashlib,json,math,shutil,os
from pathlib import Path
import torch,numpy as np
from .data import sample_starts
from .cache import FeatureCache
from concurrent.futures import ThreadPoolExecutor

class Runtime:
 def __init__(self,base,checkpoint,template_path,device='cuda',checkpointing=True):
  import transformers.utils.import_utils as iu,transformers.utils as tu
  iu.is_torchao_available=lambda *a,**kw:False;tu.is_torchao_available=iu.is_torchao_available
  from transformers import AutoModelForImageTextToText,AutoProcessor
  from peft import PeftModel
  import peft.import_utils as piu,peft.tuners.lora.torchao as pta
  piu.is_torchao_available=lambda *a,**kw:False;pta.is_torchao_available=piu.is_torchao_available
  from safetensors.torch import load_file
  self.device=device;self.proc=AutoProcessor.from_pretrained(base)
  self.full=AutoModelForImageTextToText.from_pretrained(base,dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
  self.full.requires_grad_(False);self.full.eval()
  self.model=PeftModel.from_pretrained(self.full.model.language_model,checkpoint+'/adapter',is_trainable=True)
  self.full.model.language_model=self.model
  conf=self.model.peft_config['default'];assert (conf.r,conf.lora_alpha,conf.lora_dropout)==(8,16,0)
  published=load_file(checkpoint+'/adapter/adapter_model.safetensors')
  for n,p in self.model.named_parameters():
   if p.requires_grad:
    assert 'lora_' in n;torch.testing.assert_close(p.detach().cpu(),published[n.replace('.default.','.')].float(),rtol=0,atol=0)
  self.head=torch.nn.Linear(2048,1,dtype=torch.float32).to(device);self.head.load_state_dict(torch.load(checkpoint+'/head.pt',map_location=device,weights_only=True))
  self.set_checkpointing(checkpointing)
  self.template={k:v[:1].to(device) for k,v in torch.load(template_path,map_location='cpu',weights_only=True).items() if isinstance(v,torch.Tensor)}
  t=self.template;assert t['input_ids'].shape==(1,330)
  self.emb=self.full.model.get_input_embeddings()(t['input_ids']).detach()
  self.mask=(t['input_ids']==self.full.config.video_token_id)[0];assert self.mask.sum()==196
  self.pos=self.full.model.compute_3d_position_ids(input_ids=t['input_ids'],inputs_embeds=self.emb,video_grid_thw=t['video_grid_thw'],attention_mask=t['attention_mask'],mm_token_type_ids=t['mm_token_type_ids'])
 def set_checkpointing(self,enabled):
  if enabled:self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  else:self.model.gradient_checkpointing_disable()
  self.checkpointing=enabled
 def forward(self,visual):
  b=len(visual);e=self.emb.expand(b,-1,-1).clone();e[:,self.mask]=visual.to(self.device,dtype=e.dtype,non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   h=self.model(inputs_embeds=e,position_ids=self.pos.expand(-1,b,-1),attention_mask=self.template['attention_mask'].expand(b,-1),use_cache=False,return_dict=True).last_hidden_state[:,-1]
  return self.head(h.float()).flatten()
 @torch.no_grad()
 def vision(self,windows):
  data=self.proc.video_processor(videos=list(windows),do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
  value=self.full.model.get_video_features(data['pixel_values_videos'].to(self.device),data['video_grid_thw'].to(self.device),return_dict=True).pooler_output
  value=torch.cat(value) if isinstance(value,(list,tuple)) else value
  return value.reshape(len(windows),196,2048).cpu()

class VisualStore:
 def __init__(self,runtime,root,local_cache='/content/dss_feature_cache'):
  self.runtime=runtime;self.root=Path(root);(self.root/'window_cache').mkdir(exist_ok=True)
  self.cache=FeatureCache(local_cache);self.metadata={};self.prefetcher=ThreadPoolExecutor(max_workers=1)
  self.pending={}
  self.fingerprint={'vision':'Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc','precision':'bf16','frames':8,'fps':2,'pixels':224,'canonical':'mp4v full-video round-trip'}
 def descriptor(self,record,epoch,exhaustive=False):
  name=record['name']
  if name not in self.metadata:self.metadata[name]=json.loads((self.root/'metadata'/(name+'.json')).read_text())
  m=self.metadata[name]
  starts=list(range(m['window_count'])) if exhaustive else sample_starts(m['window_count'],epoch,name)
  signature={**self.fingerprint,'source':m['sha256'],'canonical':m['canonical_sha256'],'starts':starts}
  key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();p=self.root/'window_cache'/(key+'.pt')
  return m,starts,signature,p
 def prefetch(self,records,epoch):
  # CPU-only reads of ALREADY persisted features. Never run GPU vision in a thread.
  self.pending={k:v for k,v in self.pending.items() if not v.done()}
  for record in records:
   _,_,signature,path=self.descriptor(record,epoch)
   if path.exists() and str(path) not in self.pending:
    self.pending[str(path)]=self.prefetcher.submit(self.cache.get,path,signature)
 def __call__(self,record,epoch,exhaustive=False):
  import cv2
  name=record['name'];m,starts,signature,p=self.descriptor(record,epoch,exhaustive)
  if str(p) in self.pending:
   value=self.pending.pop(str(p)).result()
   if value is not None:return value
  cached=self.cache.get(p,signature)
  if cached is not None:return cached
  # Read locally; Drive persistence and fingerprints survive runtime replacement.
  local=Path('/content/dss_canonical')/name;local.parent.mkdir(exist_ok=True);shutil.copyfile(self.root/'canonical'/name,local)
  assert hashlib.sha256(local.read_bytes()).hexdigest()==m['canonical_sha256']
  valid=m['sampled_frames']
  if exhaustive:iterator=iter_canonical_windows(local,valid,starts,4)
  else:
   frames=decode_canonical_frames(local,valid,starts,False)
   iterator=((i,[np.stack([frames[min(2*s+j,valid)] for j in range(8)]) for s in starts[i:i+4]]) for i in range(0,len(starts),4))
  features=None
  try:
   for i,windows in iterator:
    if hasattr(self,'guard'):self.guard()
    value=self.runtime.vision(windows)
    if features is None:features=torch.empty((len(starts),*value.shape[1:]),dtype=value.dtype,device='cpu')
    features[i:i+len(value)].copy_(value)
  finally:
   if hasattr(iterator,'close'):iterator.close()
   local.unlink(missing_ok=True)
  assert features is not None
  self.cache.put(p,signature,features);return features



def decode_canonical_frames(path,valid,starts,exhaustive=False):
    """Exact decoded pixels; sparse seeks for sampled training bags only."""
    import cv2
    assert valid>=1 and starts
    cap=cv2.VideoCapture(str(path));assert cap.isOpened()
    try:
        if exhaustive:
            frames=[]
            while True:
                ok,f=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            assert len(frames)==valid+1
        else:
            frames={}
            wanted=sorted({min(2*s+j,valid) for s in starts for j in range(8)})
            for i in wanted:
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                ok,f=cap.read();assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES))==i+1
                frames[i]=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        return frames
    finally:cap.release()


def iter_canonical_windows(path,valid,starts,batch=4):
    """Sequential 1-second overlap without retaining the whole decoded video."""
    import cv2
    assert starts and starts==sorted(starts) and batch>=1
    cap=cv2.VideoCapture(str(path));assert cap.isOpened();cursor=0;frames={}
    try:
        for offset in range(0,len(starts),batch):
            selected=starts[offset:offset+batch];first=min(2*selected[0],valid)
            wanted=min(2*selected[-1]+7,valid)
            while cursor<=wanted:
                ok,f=cap.read();assert ok
                if cursor>=first:frames[cursor]=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
                cursor+=1
            yield offset,[np.stack([frames[min(2*s+j,valid)] for j in range(8)]) for s in selected]
            if offset+batch<len(starts):
                cut=min(2*starts[offset+batch],valid)
                frames={k:v for k,v in frames.items() if k>=cut}
        assert cursor==valid+1 and not cap.read()[0]
    finally:cap.release()
