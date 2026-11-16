import json,os,sys,time,shutil,subprocess,traceback,hashlib
from pathlib import Path
START=float(os.environ.get('VISUAL_RECOVERY_STARTED',time.time()))
DEADLINE=START+8*60
ROOT=Path('/content/drive/MyDrive/OpenJev_UCFCrime');TRAIN=ROOT/'weak_dss_learning_10cu_v1';FORMAL=ROOT/'formal_stride1';OUT=TRAIN/'finetuned_test_raw_v1'
STATE={'stage':'setup','started':START,'status':'running','generated':0,'existing':0}
def persist():
 STATE['elapsed_s']=time.time()-START
 (OUT/'cache_recovery_status.json').write_text(json.dumps(STATE,indent=2))
 print('CACHE_RECOVERY',STATE['stage'],STATE.get('video',''),STATE['generated'],STATE['existing'],round(STATE['elapsed_s']),flush=True)
def guard():
 if time.time()>DEADLINE-60:raise TimeoutError('Visual recovery reserve reached; partial cache remains safely in Drive')
try:
 dep=Path('/content/visual_recovery_deps');dep.mkdir(exist_ok=True)
 subprocess.run([sys.executable,'-m','pip','install','-q','--target',str(dep),'--upgrade','--no-deps','transformers==5.10.2','peft==0.19.1','av==19.0.0','einops==0.8.2','fla-core==0.5.2','tokenizers==0.22.2'],check=True,timeout=180)
 sys.path.insert(0,str(dep));sys.path.insert(0,str(TRAIN))
 import torch
 from openjev_dss.runtime import Runtime,VisualStore,iter_canonical_windows
 torch.set_num_threads(2);assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'
 STATE['gpu']=torch.cuda.get_device_name(0);STATE['torch']=torch.__version__;persist();guard()
 cfg=json.loads((FORMAL/'config.json').read_text());recs=json.loads((FORMAL/'manifest.json').read_text())['videos'];canon={r['name']:r for r in cfg['canonical']}
 root=Path('/content/visual_recovery_test');root.mkdir(exist_ok=True);(root/'metadata').mkdir(exist_ok=True)
 (root/'canonical').symlink_to(FORMAL/'canonical',target_is_directory=True)
 (root/'window_cache').symlink_to(OUT/'visual_cache',target_is_directory=True)
 for r in recs:
  m=canon[r['name']];assert m['source_sha256']==r['sha256']
  (root/'metadata'/(r['name']+'.json')).write_text(json.dumps({**m,'sha256':r['sha256'],'n_frames':r['n_frames'],'fps':r['fps']}))
 # Descriptor computation does not need a loaded model.
 provider=VisualStore(None,root,local_cache='/content/visual_recovery_scratch')
 missing=[];index=[]
 for r in recs:
  m,starts,sig,p=provider.descriptor(r,0,True)
  index.append({'video':r['name'],'cache_file':p.name,'windows':len(starts),'signature':sig})
  if not p.exists():missing.append(r)
  else:STATE['existing']+=1
 STATE.update(stage='load_frozen_vision',missing_videos=len(missing),missing_windows=sum(canon[r['name']]['window_count'] for r in missing));persist();guard()
 base=ROOT/'models/base/15852e8c16360a2fea060d615a32b45270f8a8fc';original=ROOT/'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint'
 local=Path('/content/recovery_base');shutil.copytree(base,local,dirs_exist_ok=True)
 rt=Runtime(str(local),str(original),str(FORMAL/'template.pt'),checkpointing=False);provider.runtime=rt;provider.guard=guard
 # Fixed 16 temporal pairs have the same flattened vision batch shape as four 8-frame windows.
 import cv2,numpy as np
 @torch.no_grad()
 def build_visual(r):
  m,starts,sig,p=provider.descriptor(r,0,True);n=len(starts);valid=m['sampled_frames']
  video=Path('/content/recovery_video.mp4');shutil.copyfile(FORMAL/'canonical'/r['name'],video)
  assert hashlib.sha256(video.read_bytes()).hexdigest()==m['canonical_sha256']
  cap=cv2.VideoCapture(str(video));frames=[]
  try:
   while True:
    ok,f=cap.read()
    if not ok:break
    frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
  finally:cap.release()
  assert len(frames)==valid+1
  pairs=torch.empty((n+1,49,2048),dtype=torch.bfloat16)
  for offset in range(0,n+1,16):
   guard();batch=[np.stack([frames[min(2*j,valid)],frames[min(2*j+1,valid)]]) for j in range(offset,min(n+1,offset+16))]
   actual=len(batch);batch += [batch[-1]]*(16-actual)
   inp=rt.proc.video_processor(videos=batch,do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
   assert inp['video_grid_thw'].tolist()==[[1,14,14]]*16
   value=rt.full.model.get_video_features(inp['pixel_values_videos'].cuda(),inp['video_grid_thw'].cuda(),return_dict=True).pooler_output
   value=torch.cat(value) if isinstance(value,(list,tuple)) else value
   pairs[offset:offset+actual].copy_(value.reshape(16,49,2048)[:actual].cpu())
  ix=(torch.arange(n)[:,None]+torch.arange(4)[None,:]).clamp(max=n)
  features=pairs[ix].reshape(n,196,2048)
  # Preserve the exact original shape of any partial final four-window batch.
  tail=n%4
  if tail:
   windows=[np.stack([frames[min(2*t+j,valid)] for j in range(8)]) for t in range(n-tail,n)]
   features[-tail:].copy_(rt.vision(windows))
  video.unlink();return features
 # Check every window of three retained short videos, including overlap and partial tails.
 gates=[r for r in recs if canon[r['name']]['window_count']<=32 and provider.descriptor(r,0,True)[3].exists()][:3]
 checks=[]
 for gate in gates:
  m,starts,sig,p=provider.descriptor(gate,0,True)
  cache_local=Path('/content/recovery_gate.pt');shutil.copyfile(p,cache_local)
  cached=torch.load(cache_local,map_location='cpu',weights_only=True,mmap=True);assert cached['signature']==sig
  fresh=build_visual(gate)
  torch.testing.assert_close(fresh,cached['features'],rtol=0,atol=0)
  checks.append({'video':gate['name'],'windows':len(fresh),'bit_identical':True})
 STATE['gate']={'method':'16 temporal pairs plus exact partial-tail batches','checks':checks,'bit_identical':True};persist()
 for r in missing:
  guard();STATE.update(stage='recovering',video=r['name']);persist()
  m,starts,sig,p=provider.descriptor(r,0,True)
  if not p.exists():
   f=build_visual(r)
   for chunk in f.split(64):assert torch.isfinite(chunk).all()
   provider.cache.put(p,sig,f)
  STATE['generated']+=1;persist()
 for row in index:
  p=OUT/'visual_cache'/row['cache_file'];assert p.exists();row['bytes']=p.stat().st_size
 (OUT/'visual_cache_manifest.json').write_text(json.dumps({'records':index,'unique_files':len({r['cache_file'] for r in index}),'total_bytes':sum({r['cache_file']:r['bytes'] for r in index}.values()),'precision':'bf16','same_frozen_vision_gate':STATE['gate'],'created_at':time.time()},indent=2))
 STATE.update(status='complete',stage='done');persist()
except BaseException:
 STATE.update(status='error',error=traceback.format_exc());persist();raise
