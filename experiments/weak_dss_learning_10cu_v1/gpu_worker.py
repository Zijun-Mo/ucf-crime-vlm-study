import os,sys,time,json,shutil,traceback,threading
from pathlib import Path
ROOT=Path('/content/drive/MyDrive/OpenJev_UCFCrime');OUT=ROOT/'weak_dss_learning_10cu_v1'
DEADLINE=float(os.environ['DSS_LEARNING_DEADLINE']);HARD=float(os.environ['DSS_LEARNING_HARD_STOP'])
STATUS={'status':'running','stage':'copy','started':time.time()}
def persist():
 (OUT/'gpu_setup_status.json').write_text(json.dumps(STATUS));Path('/content/dss_learning_setup_status.json').write_text(json.dumps(STATUS))
def stopwatch():
 while time.time()<HARD:time.sleep(min(15,HARD-time.time()))
 try:
  from google.colab import runtime
  runtime.unassign()
 finally:os._exit(0)
threading.Thread(target=stopwatch,daemon=True).start()
try:
 import torch,hashlib
 from openjev_dss.runtime import Runtime,VisualStore
 from openjev_dss.learning import run,AllocationExpired
 from openjev_dss.kernels import restore_kernel_cache,save_kernel_cache
 torch.set_num_threads(2);persist();work=Path('/content/dss_learning');work.mkdir(exist_ok=True)
 def guard():
  if time.time()>DEADLINE-150:raise AllocationExpired('Feature extraction budget deadline')
 def cp(src,dst):
  guard();shutil.copyfile(src,dst);return dst
 for src,dst in [(ROOT/'models/base/15852e8c16360a2fea060d615a32b45270f8a8fc',work/'base'),(ROOT/'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint',work/'checkpoint')]:
  shutil.copytree(src,dst,dirs_exist_ok=True,copy_function=cp)
 STATUS['stage']='model';persist();guard()
 rt=Runtime(str(work/'base'),str(work/'checkpoint'),str(ROOT/'formal_stride1/template.pt'))
 STATUS['compiled_cache']=restore_kernel_cache(ROOT/'weak_dss_efficiency_v1/kernel_cache')
 provider=VisualStore(rt,ROOT/'weak_dss_smoke_v1');provider.guard=guard
 manifest=json.loads((OUT/'manifest.json').read_text())
 def frozen_hash():
  h=hashlib.sha256()
  for n,p in rt.full.named_parameters():
   if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 STATUS['frozen_before']=frozen_hash();STATUS['stage']='gradient_gate';persist()
 from openjev_dss.losses import loss
 from openjev_dss.training import two_pass
 selected=[next(r for r in manifest['records'] if r['split']=='train' and r['label']==y) for y in [0,1]]
 fixture=[provider(r,0,False) for r in selected]
 params=[p for p in rt.model.parameters() if p.requires_grad]+list(rt.head.parameters())
 rt.model.train();rt.head.train()
 packed=torch.cat(fixture);scores=torch.cat([rt.forward(packed[i:i+4]).float() for i in range(0,len(packed),4)])
 objective,_=loss(list(scores.split([len(v) for v in fixture])),[0,1],True);objective.backward()
 dense=torch.cat([p.grad.detach().float().flatten() for p in params]);rt.model.zero_grad(set_to_none=True);rt.head.zero_grad(set_to_none=True)
 two_pass(rt.forward,fixture,[0,1],True,chunk=4,score_batch=4,pad_batches=True)
 leaf=torch.cat([p.grad.detach().float().flatten() for p in params]);relative=float(torch.linalg.vector_norm(dense-leaf)/torch.linalg.vector_norm(dense));cosine=float(torch.nn.functional.cosine_similarity(dense.double(),leaf.double(),dim=0))
 STATUS['gradient_gate']={'relative_l2':relative,'cosine':cosine,'finite':bool(torch.isfinite(leaf).all()),'windows':len(packed),'criteria':'relative L2 < 1%; cosine > .999'}
 assert relative<.01 and cosine>.999 and torch.isfinite(leaf).all(),STATUS['gradient_gate']
 rt.model.zero_grad(set_to_none=True);rt.head.zero_grad(set_to_none=True)
 del scores,objective,dense,leaf;persist()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  rt.forward(packed[:4]).sum().backward()
 STATUS['operators']=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and any(k in e.name.lower() for k in ['chunk','gated_delta','flash','attention'])})
 STATUS['fast_linear_attention_seen']=any('chunk' in n.lower() or 'gated_delta' in n.lower() for n in STATUS['operators'])
 rt.model.zero_grad(set_to_none=True);rt.head.zero_grad(set_to_none=True);persist()
 STATUS.update(stage='learning',gpu=torch.cuda.get_device_name(),torch=torch.__version__,deadline=DEADLINE);persist()
 from openjev_dss.training import trainable_state
 initial_trainable=trainable_state(rt.model,rt.head)
 result=run(rt,manifest,provider,OUT/'training',DEADLINE,epochs=int(os.environ.get('DSS_LEARNING_EPOCHS','3')))
 STATUS.update(status=result['status'],stage='finished',baseline=result['baseline'],arms={a:{'steps':len(v['rows']),'best_ap':v['best_ap']} for a,v in result['arms'].items()})
 # Independent export reload check and immutable base digest for the report.
 STATUS['exported_arms']=[];STATUS['trainable_update_l2']={}
 for arm in ['A','B']:
  folder=OUT/'training'/arm
  if (folder/'best_state.pt').exists():
   from openjev_dss.training import restore
   from safetensors.torch import load_file
   state=torch.load(folder/'best_state.pt',map_location='cpu',weights_only=True)
   published=load_file(str(folder/'best_adapter/adapter_model.safetensors'))
   STATUS['exported_arms'].append(arm)
   STATUS['trainable_update_l2'][arm]={part:float(sum((v-initial_trainable[k]).double().square().sum() for k,v in state.items() if k.startswith(part+'.')).sqrt()) for part in ['model','head']}
   assert all(v>0 for v in STATUS['trainable_update_l2'][arm].values())
   for name,value in state.items():
    if name.startswith('model.'):
     key=name[len('model.'):].replace('.default.','.')
     torch.testing.assert_close(value.float(),published[key].float(),rtol=0,atol=0)
   hs=torch.load(folder/'best_head.pt',map_location='cpu',weights_only=True)
   for key,value in hs.items():torch.testing.assert_close(value,state['head.'+key],rtol=0,atol=0)
 STATUS['export_tensor_restore_verified']=True
 STATUS['frozen_after']=frozen_hash();STATUS['frozen_hash_unchanged']=STATUS['frozen_before']==STATUS['frozen_after'];assert STATUS['frozen_hash_unchanged']
except Exception:STATUS.update(status='error',error=traceback.format_exc())
finally:
 STATUS['elapsed_s']=time.time()-STATUS['started'];persist()
 try:save_kernel_cache(OUT/'kernel_cache')
 except Exception:pass
