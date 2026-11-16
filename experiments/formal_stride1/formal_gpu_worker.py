import time,json,sys,os,traceback,shutil,copy,gc,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,zipfile,importlib.metadata,inspect
ROOT=Path('/content/drive/MyDrive/OpenJev_UCFCrime');OUT=ROOT/'formal_stride1'
DEADLINE=1791730514.6599967
GS={'status':'running','started':time.time(),'stage':'packages','features_videos':0,'models':{},'errors':[]}
def atomic(path,value):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def persist():
 Path('/content/formal_gpu_status.json').write_text(json.dumps(GS,indent=2))
 try:atomic(OUT/'gpu_status.json',GS)
 except OSError as e:print('DRIVE_STATUS_WRITE_WARNING',repr(e),flush=True)
def guard(reserve=0):
 if time.time()>DEADLINE-reserve:raise TimeoutError('Formal test GPU budget reserve reached')
def run():
 try:
  dep=Path('/content/formal_deps');dep.mkdir(exist_ok=True)
  subprocess.run([sys.executable,'-m','pip','install','-q','--target',str(dep),'--upgrade','--no-deps','transformers==5.10.2','peft==0.19.1','av==19.0.0','einops==0.8.2','fla-core==0.5.2','tokenizers==0.22.2'],check=True,timeout=240)
  sys.path.insert(0,str(dep))
  import transformers.utils.import_utils as import_utils
  import transformers.utils as transformer_utils
  # Match prior FP16-only smoke environment without uninstalling Colab system packages.
  import_utils.is_torchao_available=lambda *args,**kwargs:False
  transformer_utils.is_torchao_available=import_utils.is_torchao_available
  import torch,numpy as np,cv2
  from transformers import AutoModelForImageTextToText,AutoProcessor
  import transformers.models.qwen3_5.modeling_qwen3_5 as qm
  from sklearn.metrics import roc_auc_score,average_precision_score
  torch.set_num_threads(2);cv2.setNumThreads(1)
  cfg=json.loads((OUT/'config.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());records=manifest['videos'];canon={x['name']:x for x in cfg['canonical']};finger=cfg['fingerprint']
  assert len(records)==290 and sum(canon[r['name']]['window_count'] for r in records)==37188
  GS.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,dependency_isolation=str(dep),torchao_integration_disabled=True,versions={n:importlib.metadata.version(n) for n in ['transformers','fla-core','triton','peft','av','tokenizers','einops']},fingerprint=finger,deadline=DEADLINE,vision_pairs_per_batch=4);persist()
  work=Path('/content/formal_gpu');work.mkdir(exist_ok=True)
  features=OUT/'features';features.mkdir(exist_ok=True)
  for mode in ['native_qwen','merged_openjev']:(OUT/mode).mkdir(exist_ok=True)
  counts={}
  def instrument(model):
   for mod in model.model.language_model.modules():
    if isinstance(mod,qm.Qwen3_5GatedDeltaNet):
     fn=mod.chunk_gated_delta_rule;label=fn.__module__+'.'+fn.__name__
     assert label.startswith('fla.'),('Expected FLA chunk kernel',label)
     def counted(*a,_fn=fn,_label=label,**kw):
      counts[_label]=counts.get(_label,0)+1;return _fn(*a,**kw)
     mod.chunk_gated_delta_rule=counted
  def copy_safe(src,dst):
   src=Path(src);dst=Path(dst)
   for attempt in range(5):
    guard(120)
    try:
     expected=src.stat().st_size
     if dst.exists() and dst.stat().st_size==expected:return str(dst)
     with src.open('rb') as reader,dst.open('wb') as writer:
      shutil.copyfileobj(reader,writer,length=4*1024*1024)
     assert dst.stat().st_size==expected,(src,'size mismatch')
     return str(dst)
    except OSError as e:
     print('DRIVE_COPY_RETRY',str(src),attempt,repr(e),flush=True)
     if attempt==4:raise
     time.sleep(10)
  GS['stage']='copy_base';persist();guard(120)
  shutil.copytree(cfg['base'],work/'base',dirs_exist_ok=True,copy_function=copy_safe)
  proc=AutoProcessor.from_pretrained(str(work/'base'));proc.tokenizer.padding_side='right'
  full=AutoModelForImageTextToText.from_pretrained(str(work/'base'),dtype=torch.float16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
  instrument(full)
  template=torch.load(OUT/'template.pt',map_location='cpu',weights_only=True)
  for k in ['input_ids','attention_mask','mm_token_type_ids','video_grid_thw']:
   assert bool((template[k]==template[k][:1]).all()),('template not constant',k)
   template[k]=template[k][:1]
  assert template['input_ids'].shape==(1,330)
  early=proc.video_processor(videos=[np.zeros((8,224,224,3),dtype=np.uint8)],do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
  assert early['video_grid_thw'].tolist()==[[4,14,14]]
  atomic(OUT/'processor_test.json',{'class':type(proc.video_processor).__name__,'shapes':{k:list(v.shape) for k,v in early.items()},'settings':proc.video_processor.to_dict()})
  (OUT/'video_processor_reference.py').write_text(inspect.getsource(type(proc.video_processor)))
  def read_frames(rec):
   src=work/('canonical_'+rec['name']);shutil.copyfile(OUT/'canonical'/rec['name'],src)
   assert hashlib.sha256(src.read_bytes()).hexdigest()==canon[rec['name']]['canonical_sha256']
   cap=cv2.VideoCapture(str(src));arr=[]
   while True:
    ok,bgr=cap.read()
    if not ok:break
    arr.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   cap.release();src.unlink();assert len(arr)==canon[rec['name']]['canonical_frames'];return np.stack(arr)
  def pixels(frames):return proc.video_processor(videos=[frames],do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
  @torch.inference_mode()
  def vision(inp):
   o=full.model.get_video_features(inp['pixel_values_videos'].cuda(),inp['video_grid_thw'].cuda(),return_dict=True).pooler_output
   return torch.cat(o) if isinstance(o,(tuple,list)) else o
  def window_frames(frames,rec,start):
   valid=canon[rec['name']]['sampled_frames'];padding=frames[valid]
   return np.stack([frames[start*2+j] if start*2+j<valid else padding for j in range(8)])
  @torch.inference_mode()
  def make_features(frames,rec):
   valid=canon[rec['name']]['sampled_frames'];nblocks=canon[rec['name']]['window_count'];pad=frames[valid];blocks=[]
   for i in range(nblocks):blocks.append(np.stack([frames[2*i],frames[2*i+1] if 2*i+1<valid else pad]))
   blocks.append(np.stack([pad,pad]));results=[]
   for i in range(0,len(blocks),4):
    guard(120);batch=blocks[i:i+4];actual=len(batch);batch=batch+[batch[-1]]*(4-actual)
    vi=proc.video_processor(videos=batch,do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
    assert bool((vi['video_grid_thw'][:,0]==1).all()) and bool((vi['video_grid_thw'][:,1:]==14).all())
    results.append(vision(vi).reshape(-1,49,2048)[:actual].cpu())
   return torch.cat(results)
  def select_features(f,start):
   pad=len(f)-1;return f[[min(start+j,pad) for j in range(4)]].reshape(196,2048)
  def clone(cache,b):
   result=copy.copy(cache);result.layers=[]
   for l in cache.layers:
    new=copy.copy(l)
    for k,v in vars(l).items():
     if isinstance(v,torch.Tensor):setattr(new,k,v.repeat_interleave(b,0).clone())
    if hasattr(new,'max_batch_size'):new.max_batch_size=b
    result.layers.append(new)
   return result
  # Same canonical input gate: cross-block boundary, overlap, odd final pair and padded tail.
  GS['stage']='cache_gate';persist();checks=[]
  gate_names=['Abuse028_x264.mp4','Arson009_x264.mp4','Normal_Videos_881_x264.mp4']
  with torch.inference_mode():
   for name in gate_names:
    rec=next(x for x in records if x['name']==name);frames=read_frames(rec);f=make_features(frames,rec);n=canon[name]['window_count']
    for start in sorted(set([0,1,max(0,n-4),n-1])):
     vi=pixels(window_frames(frames,rec,start));assert torch.equal(vi['video_grid_thw'],template['video_grid_thw'])
     wf=window_frames(frames,rec,start)
     packed=proc.video_processor(videos=[wf[j:j+2] for j in range(0,8,2)],do_sample_frames=False,return_tensors='pt',input_data_format='channels_last')
     assert torch.equal(packed['pixel_values_videos'],vi['pixel_values_videos']),'Temporal-pair preprocessing differs from full window'
     direct=vision(vi);cached=select_features(f,start).cuda();vd=float((direct-cached).abs().max())
     data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in template.items()}
     embeds=full.model.get_input_embeddings()(data['input_ids']);_,vm=full.model.get_placeholder_mask(data['input_ids'],inputs_embeds=embeds,video_features=direct)
     pos=full.model.compute_3d_position_ids(input_ids=data['input_ids'],inputs_embeds=embeds,video_grid_thw=data['video_grid_thw'],attention_mask=data['attention_mask'],mm_token_type_ids=data['mm_token_type_ids'])
     e1=embeds.masked_scatter(vm,direct);e2=embeds.masked_scatter(vm,cached)
     h1=full.model.language_model(inputs_embeds=e1,position_ids=pos,attention_mask=data['attention_mask'],use_cache=False).last_hidden_state[:,-1].float()
     h2=full.model.language_model(inputs_embeds=e2,position_ids=pos,attention_mask=data['attention_mask'],use_cache=False).last_hidden_state[:,-1].float()
     w=full.lm_head.weight[[9175,2665]].float();l1=torch.nn.functional.linear(h1,w);l2=torch.nn.functional.linear(h2,w);s1=l1[:,0]-l1[:,1];s2=l2[:,0]-l2[:,1]
     pd=float((s1.sigmoid()-s2.sigmoid()).abs().max());checks.append({'video':name,'start_s':start,'max_feature_abs_diff':vd,'pixel_values_identical':True,'max_feature_magnitude':float(direct.abs().max()),'native_logit_diff':float((s1-s2).abs().max()),'native_probability_diff':pd})
     assert vd<=0.05 and pd<=0.005,('visual cache gate failed',checks[-1])
    target=features/(name+'.pt');torch.save({'features':f,'fingerprint':finger,'source_sha256':rec['sha256'],'canonical_sha256':canon[name]['canonical_sha256']},target)
   atomic(OUT/'cache_gate.json',{'passed':True,'checks':checks,'scope':'same canonical decoded pixels; independent temporal-patch vision features vs full window; FP16 numerical tolerance; no labels used','preprocess':cfg['preprocess']})
   GS['cache_gate_passed']=True;GS['stage']='visual_features';persist()
   # Per-video artifacts are durable and resumable, including a padding-pair feature.
   with ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(read_frames,records[0])
    for i,rec in enumerate(records):
     guard(120);frames=future.result()
     if i+1<len(records):future=pool.submit(read_frames,records[i+1])
     target=features/(rec['name']+'.pt')
     if not target.exists():
      f=make_features(frames,rec);torch.save({'features':f,'fingerprint':finger,'source_sha256':rec['sha256'],'canonical_sha256':canon[rec['name']]['canonical_sha256']},target)
     GS['features_videos']=i+1;GS['elapsed_s']=time.time()-GS['started'];persist()
   GS['feature_bytes']=sum(f.stat().st_size for f in features.glob('*.pt'));GS['features_complete']=time.time();persist()
   for mode in ['native_qwen','merged_openjev']:
    if mode=='merged_openjev':
     del full;gc.collect();torch.cuda.empty_cache();GS['stage']='copy_merged';persist();guard(120)
     shutil.copytree(cfg['merged'],work/'merged',dirs_exist_ok=True,copy_function=copy_safe)
     full=AutoModelForImageTextToText.from_pretrained(str(work/'merged'),dtype=torch.float16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
     assert not any('lora_' in k for k in full.state_dict())
     instrument(full)
     head=torch.nn.Linear(2048,1,dtype=torch.float32).cuda().eval();head.load_state_dict(torch.load(work/'merged/jev_head/head.pt',map_location='cuda',weights_only=True))
     temp=json.loads((work/'merged/jev_head/temperature.json').read_text())['temperature']
    lm=full.model.language_model
    data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in template.items()}
    emb=full.model.get_input_embeddings()(data['input_ids']);prefix=int((data['input_ids'][0]==full.config.vision_start_token_id).nonzero()[0]);assert prefix==91
    placeholder=(data['input_ids']==full.config.video_token_id)[0];assert int(placeholder.sum())==196
    pos=full.model.compute_3d_position_ids(input_ids=data['input_ids'],inputs_embeds=emb,video_grid_thw=data['video_grid_thw'],attention_mask=data['attention_mask'],mm_token_type_ids=data['mm_token_type_ids'])
    pre=lm(inputs_embeds=emb[:,:prefix],position_ids=pos[:,:,:prefix],attention_mask=data['attention_mask'][:,:prefix],use_cache=True,return_dict=True).past_key_values
    saved=[v.clone() for l in pre.layers for v in vars(l).values() if isinstance(v,torch.Tensor)]
    rowweight=full.lm_head.weight[[9175,2665]].float() if mode=='native_qwen' else None
    @torch.inference_mode()
    def score(visual):
     b=len(visual);e=emb.expand(b,-1,-1).clone();e[:,placeholder]=visual;mask=data['attention_mask'].expand(b,-1)
     h=lm(inputs_embeds=e[:,prefix:],position_ids=pos[:,:,prefix:].expand(-1,b,-1),attention_mask=mask,past_key_values=clone(pre,b),use_cache=True,return_dict=True).last_hidden_state[:,-1].float()
     if mode=='native_qwen':
      logits=torch.nn.functional.linear(h,rowweight);return logits[:,0]-logits[:,1]
     return head(h).flatten()
    # Check both models' complete cached path against uncached full-window inference.
    prefix_checks=[]
    for name in gate_names:
     rec=next(r for r in records if r['name']==name);frames=read_frames(rec)
     f=torch.load(features/(name+'.pt'),map_location='cpu',weights_only=True)['features']
     for start in sorted(set([0,1,max(0,canon[name]['window_count']-4),canon[name]['window_count']-1])):
      direct=vision(pixels(window_frames(frames,rec,start)))
      e=emb.clone();e[:,placeholder]=direct
      h=lm(inputs_embeds=e,position_ids=pos,attention_mask=data['attention_mask'],use_cache=False).last_hidden_state[:,-1].float()
      if mode=='native_qwen':
       ls=torch.nn.functional.linear(h,rowweight);uncached=ls[:,0]-ls[:,1]
      else:uncached=head(h).flatten()
      cached=score(select_features(f,start)[None].cuda())
      delta=float((cached-uncached).abs().max());pd=float((torch.sigmoid(cached/(1 if mode=='native_qwen' else temp))-torch.sigmoid(uncached/(1 if mode=='native_qwen' else temp))).abs().max())
      prefix_checks.append({'video':name,'start_s':start,'logit_delta':delta,'probability_delta':pd})
      assert delta<=0.08 and pd<=0.01,('Complete cache gate failed',mode,prefix_checks[-1])
    atomic(OUT/(mode+'_prefix_gate.json'),{'passed':True,'checks':prefix_checks,'logit_tolerance':0.08,'probability_tolerance':0.01,'uses_test_labels':False})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
     score(select_features(f,0)[None].cuda());torch.cuda.synchronize()
    atomic(OUT/(mode+'_kernel_profile.json'),{'operators':[e.key for e in profile.key_averages()],'fla_calls':dict(counts),'transformers_fast_path_flag':bool(qm.is_fast_path_available)})
    GS['models'][mode]={'videos':0,'windows':0,'started':time.time()};GS.update(stage='inference',model=mode);persist()
    def read_features(rec):
     local=work/(rec['name']+'.pt');shutil.copyfile(features/(rec['name']+'.pt'),local)
     value=torch.load(local,map_location='cpu',weights_only=True);local.unlink();assert value['fingerprint']==finger and value['source_sha256']==rec['sha256'];return value['features']
    with ThreadPoolExecutor(max_workers=1) as pool:
     future=pool.submit(read_features,records[0])
     for i,rec in enumerate(records):
      guard(90);f=future.result()
      if i+1<len(records):future=pool.submit(read_features,records[i+1])
      n=canon[rec['name']]['window_count'];target=OUT/mode/(rec['name']+'.json')
      if target.exists():
       old=json.loads(target.read_text());assert old['fingerprint']==finger and len(old['scores'])==n
      else:
       f=f.cuda();vals=[]
       for start in range(0,n,4):
        guard(90);indices=torch.arange(start,min(start+4,n),device='cuda')[:,None]+torch.arange(4,device='cuda')[None,:];indices=indices.clamp(max=len(f)-1)
        visual=f[indices].reshape(-1,196,2048);v=score(visual);assert bool(torch.isfinite(v).all());vals.extend(v.cpu().tolist())
       atomic(target,{'name':rec['name'],'fingerprint':finger,'model':mode,'fps':rec['fps'],'n_frames':rec['n_frames'],'source_sha256':rec['sha256'],'canonical_sha256':canon[rec['name']]['canonical_sha256'],'start_s':list(range(n)),'scores':vals,'score_type':'yes_minus_no_logit' if mode=='native_qwen' else 'jev_scalar_logit','temperature':1 if mode=='native_qwen' else temp})
      GS['models'][mode].update(videos=i+1,windows=GS['models'][mode]['windows']+n,elapsed_s=time.time()-GS['models'][mode]['started']);GS['elapsed_s']=time.time()-GS['started'];persist()
    GS['models'][mode]['complete']=time.time();assert GS['models'][mode]['windows']==37188
    after=[v for l in pre.layers for v in vars(l).values() if isinstance(v,torch.Tensor)];assert all(torch.equal(a,b) for a,b in zip(saved,after));GS['models'][mode]['prefix_immutable']=True;GS['models'][mode]['kernel_calls']=dict(counts)
    del lm,pre,emb,saved,after,score;gc.collect();persist()
   GS['stage']='metrics';persist();metrics=[]
   for mode in ['native_qwen','merged_openjev']:
    for stride in [1,2,4]:
     labels=[];all_scores=[];covered=0;windows=0
     for rec in records:
      r=json.loads((OUT/mode/(rec['name']+'.json')).read_text());assert r['fingerprint']==finger
      n=rec['n_frames'];truth=np.zeros(n,dtype=np.uint8)
      for a,b in rec['intervals']:
       if a>=0 and b>=a:truth[max(0,a-1):min(n,b)]=1
      sums=np.zeros(n,dtype=np.float64);counts_frame=np.zeros(n,dtype=np.int32)
      for start,logit in zip(r['start_s'],r['scores']):
       if start%stride:continue
       lo=round(start*rec['fps']);hi=min(n,round((start+4)*rec['fps']));assert 0<=lo<hi<=n
       sums[lo:hi]+=logit;counts_frame[lo:hi]+=1;windows+=1
      assert (counts_frame>0).all();pred=sums/counts_frame
      labels.append(truth);all_scores.append(pred);covered+=n
     y=np.concatenate(labels);p=np.concatenate(all_scores);assert len(y)==1111808 and covered==1111808 and np.isfinite(p).all()
     rate={'model':mode,'stride_s':stride,'window_s':4,'windows':windows,'videos':290,'frames':len(y),'positive_frames':int(y.sum()),'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'aggregation':'mean raw logits','fingerprint':finger};metrics.append(rate)
     np.savez_compressed(OUT/f'{mode}_stride{stride}_frame_scores.npz',scores=p,labels=y)
   atomic(OUT/'metrics.json',{'config':{k:v for k,v in cfg.items() if k!='canonical'},'metrics':metrics,'cache_gate':json.loads((OUT/'cache_gate.json').read_text())})
   GS['stage']='export_bundle';persist()
   with zipfile.ZipFile(OUT/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['config.json','manifest.json','cache_gate.json','metrics.json','processor_test.json','video_processor_reference.py']:
     z.write(OUT/name,name)
    for mode in ['native_qwen','merged_openjev']:
     for p in sorted((OUT/mode).glob('*.json')):z.write(p,str(p.relative_to(OUT)))
     for suffix in ['prefix_gate','kernel_profile']:z.write(OUT/(mode+'_'+suffix+'.json'),mode+'_'+suffix+'.json')
    for p in sorted(OUT.glob('*_frame_scores.npz')):z.write(p,p.name)
   GS['results_bundle_bytes']=(OUT/'results_bundle.zip').stat().st_size
   GS.update(status='complete',stage='done',elapsed_s=time.time()-GS['started'],metrics=metrics);persist()
 except Exception:
  error=traceback.format_exc();print(error,flush=True);GS.update(status='error',error=error,elapsed_s=time.time()-GS['started']);persist()
run()
