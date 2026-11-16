"""Budgeted, resumable, paired learning run. No test evaluation or calibration."""
import json,time,math,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .training import two_pass,trainable_state,restore
from .experiment import score_windows

class AllocationExpired(Exception):pass

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def schedule(records,epochs):
    normals=[r for r in records if r['split']=='train' and not r['label']]
    anomalies=[r for r in records if r['split']=='train' and r['label']]
    assert normals and anomalies
    result=[]
    for ep in range(epochs):
        rng=np.random.default_rng(42+ep);n=rng.permutation(len(normals));a=rng.permutation(len(anomalies))
        pairs=[(normals[n[j%len(n)]],anomalies[a[j%len(a)]]) for j in range(4*math.ceil(max(len(n),len(a))/4))]
        result.extend((ep,pairs[i:i+4]) for i in range(0,len(pairs),4))
    return result

def run(rt,manifest,provider,out,deadline,epochs=3):
    assert 1<=epochs<=3 and manifest['audit']['near_duplicate_visual_audit']=='passed'
    out=Path(out);out.mkdir(exist_ok=True);val=[r for r in manifest['records'] if r['split']=='val']
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    planned=schedule(manifest['records'],epochs)
    params=[p for p in rt.model.parameters() if p.requires_grad]+list(rt.head.parameters())
    initial=trainable_state(rt.model,rt.head)
    frozen=[(n,p,p._version) for n,p in rt.full.named_parameters() if not p.requires_grad]
    state={'status':'running','started':time.time(),'manifest_sha256':fingerprint,'epochs_target':epochs,'steps_target_per_arm':len(planned),'test_used':False,'baseline':None,'arms':{'A':{'rows':[],'best_ap':None},'B':{'rows':[],'best_ap':None}}}
    arms={}
    for arm in ['A','B']:
        folder=out/arm;folder.mkdir(exist_ok=True)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        opt=torch.optim.AdamW([{'params':[p for p in rt.model.parameters() if p.requires_grad],'lr':1e-5},{'params':list(rt.head.parameters()),'lr':1e-4}],weight_decay=.01)
        arms[arm]={'weights':copy.deepcopy(initial),'opt':opt,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    saved=out/'status.json'
    if saved.exists():
        previous=json.loads(saved.read_text());assert previous['manifest_sha256']==fingerprint
        state=previous;state['status']='running'
        for arm in ['A','B']:
            cp=torch.load(out/arm/f"resume_{state['resume_generation']%2}.pt",map_location='cpu',weights_only=True)
            arms[arm]['weights']=cp['trainable'];arms[arm]['opt'].load_state_dict(cp['optimizer']);arms[arm]['rng']=cp['rng'];arms[arm]['cuda_rng']=cp['cuda_rng']
            assert cp['steps']==len(state['arms'][arm]['rows'])
    def guard(reserve=150):
        if time.time()>deadline-reserve:raise AllocationExpired('GPU allocation deadline; paired state persisted')
    def persist():
        # Alternate two generations. Status commits only after BOTH resume files;
        # a crash while saving cannot overwrite the previously committed pair.
        generation=state.get('resume_generation',-1)+1
        for arm,part in arms.items():
            path=out/arm/f'resume_{generation%2}.pt';tmp=path.with_suffix('.tmp')
            torch.save({'trainable':part['weights'],'optimizer':part['opt'].state_dict(),'rng':part['rng'],'cuda_rng':part['cuda_rng'],'steps':len(state['arms'][arm]['rows']),'manifest_sha256':fingerprint},tmp);tmp.replace(path)
        state['resume_generation']=generation
        state['elapsed_s']=time.time()-state['started'];atomic_json(saved,state)
        for arm in arms:atomic_json(out/arm/'training.json',state['arms'][arm]['rows'])
    def validation(tag):
        rt.model.eval();rt.head.eval();path=out/(tag+'_validation.json')
        rows=json.loads(path.read_text()) if path.exists() else []
        assert [r['name'] for r in rows]==[r['name'] for r in val[:len(rows)]], 'Validation checkpoint identity changed'
        for r in val[len(rows):]:
            guard(180);features=provider(r,0,True)
            # Chunked inference permits a deadline check between physical batches.
            zs=[]
            for i in range(0,len(features),4):
                guard();zs.append(score_windows(rt.forward,features[i:i+4]))
            z=torch.cat(zs);p=z.sigmoid();q=float(p.topk(max(1,math.ceil(len(p)/8))).values.mean())
            rows.append({'name':r['name'],'label':r['label'],'probability':q,'windows':len(z)});atomic_json(path,rows)
        return float(average_precision_score([r['label'] for r in rows],[r['probability'] for r in rows]))
    try:
        if state['baseline'] is None:
            restore(rt.model,rt.head,initial);state['stage']='baseline_validation';persist()
            state['baseline']=validation('released');persist()
        first=min(len(state['arms'][a]['rows']) for a in arms)
        # Resume any epoch validation interrupted after a complete training pair.
        for arm in arms:
            rows=state['arms'][arm]['rows']
            if rows and (first==len(planned) or planned[first][0]!=planned[first-1][0]) and 'val_video_ap' not in rows[-1]:
                epoch=planned[first-1][0];restore(rt.model,rt.head,arms[arm]['weights'])
                ap=validation(f'{arm}_epoch{epoch+1}');rows[-1]['val_video_ap']=ap
                best=state['arms'][arm]['best_ap']
                if best is None or ap>best:
                    state['arms'][arm]['best_ap']=ap
                    rt.model.save_pretrained(out/arm/'best_adapter');torch.save(rt.head.state_dict(),out/arm/'best_head.pt');torch.save(arms[arm]['weights'],out/arm/'best_state.pt')
                    atomic_json(out/arm/'selection.json',{'step':first,'epoch':epoch+1,'video_ap':ap,'test_used':False})
                persist()
        epoch_started=None
        for step in range(first,len(planned)):
            if step==0 or planned[step][0]!=planned[step-1][0]:
                previous=state.get('last_complete_epoch_seconds')
                if previous and deadline-time.time()<previous*1.10+180:
                    raise AllocationExpired('Insufficient time for another complete paired epoch and validation; preserved equal completed epochs')
                epoch_started=time.time()
            epoch,micros=planned[step];guard(240)
            state.update(stage='training',epoch=epoch+1,step=step+1)
            windows=[provider(r,epoch,False) for pair in micros for r in pair]
            guard(120)
            for arm in ['A','B']:
                if len(state['arms'][arm]['rows'])>step:continue
                part=arms[arm];restore(rt.model,rt.head,part['weights'])
                torch.set_rng_state(part['rng']);torch.cuda.set_rng_state(part['cuda_rng'])
                rt.model.train();rt.head.train();opt=part['opt'];opt.zero_grad(set_to_none=True);start=time.time()
                objective,info=two_pass(rt.forward,windows,[0,1]*len(micros),arm=='B',chunk=4,score_batch=4,pad_batches=True)
                norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
                info['gradient_tensors']=sum(p.grad is not None for p in params)
                info['head_grad_norm']=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in rt.head.parameters() if p.grad is not None])))
                opt.step();part['weights']=trainable_state(rt.model,rt.head)
                part['rng']=torch.get_rng_state();part['cuda_rng']=torch.cuda.get_rng_state()
                state['arms'][arm]['rows'].append({'step':step+1,'epoch':epoch+1,'loss':objective,'stats':info,'videos':[r['name'] for pair in micros for r in pair],'grad_norm':float(norm),'seconds':time.time()-start})
            if (step+1)%10==0:persist()
            if step+1==len(planned) or planned[step+1][0]!=epoch:
                persist()
                for arm in ['A','B']:
                    restore(rt.model,rt.head,arms[arm]['weights']);state['stage']='validation_'+arm;persist()
                    ap=validation(f'{arm}_epoch{epoch+1}');row=state['arms'][arm]['rows'][-1];row['val_video_ap']=ap
                    best=state['arms'][arm]['best_ap']
                    if best is None or ap>best:
                        state['arms'][arm]['best_ap']=ap
                        rt.model.save_pretrained(out/arm/'best_adapter');torch.save(rt.head.state_dict(),out/arm/'best_head.pt');torch.save(arms[arm]['weights'],out/arm/'best_state.pt')
                        atomic_json(out/arm/'selection.json',{'step':step+1,'epoch':epoch+1,'video_ap':ap,'test_used':False})
                    atomic_json(out/arm/'training.json',state['arms'][arm]['rows']);persist()
                if epoch_started is not None:state['last_complete_epoch_seconds']=time.time()-epoch_started;persist()
        state['status']='complete'
    except AllocationExpired as e:state.update(status='budget_stopped',reason=str(e))
    except Exception:
        import traceback;state.update(status='error',error=traceback.format_exc())
    finally:
        state['frozen_versions_unchanged']=all(p._version==version for _,p,version in frozen)
        state['paired_steps_equal']=len(state['arms']['A']['rows'])==len(state['arms']['B']['rows']);persist()
    return state
