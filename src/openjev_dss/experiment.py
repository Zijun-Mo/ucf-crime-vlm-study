"""Formal A/B training and independent video calibration; test-blind selection.
Providers: windows(record, epoch, exhaustive) -> cached visual tensor (N,196,2048).
The provider must enforce frozen-vision fingerprints and the 1-second start grid.
"""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .training import two_pass,restore,trainable_state

@torch.no_grad()
def score_windows(forward,windows,batch=4):
    if batch<1 or not len(windows):raise ValueError('Invalid inference windows/batch')
    output=[]
    for i in range(0,len(windows),batch):
        w=windows[i:i+batch];n=len(w)
        if n<batch:w=torch.cat([w,w[:1].expand(batch-n,*w.shape[1:])])
        output.append(forward(w).float().flatten()[:n].cpu())
    return torch.cat(output)

@torch.no_grad()
def video_probability(forward,windows,temperature=1.,batch=4):
    logits=score_windows(forward,windows,batch)
    p=(logits/temperature).sigmoid();k=max(1,math.ceil(len(p)/8))
    return float(p.topk(k).values.mean()),logits

def validate(forward,records,provider):
    truth=[];pred=[]
    for r in records:
        assert r['split']=='val';p,_=video_probability(forward,provider(r,0,True));truth.append(r['label']);pred.append(p)
    return float(average_precision_score(truth,pred))

def train_ab(model,head,forward,manifest,provider,out,budget_seconds,epochs=3,
             score_batch=4,backward_batch=4):
    """Same seed, initialization, pair order, sampling and optimizer steps in A/B."""
    assert 1<=epochs<=3
    audit=manifest['audit']
    if audit.get('near_duplicate_visual_audit')!='passed':raise ValueError('Full visual source/near-duplicate audit not approved')
    assert budget_seconds>0
    out=Path(out);out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+budget_seconds
    records=manifest['records'];val=[r for r in records if r['split']=='val'];train=[r for r in records if r['split']=='train']
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    normals=[r for r in train if not r['label']];anomalies=[r for r in train if r['label']]
    base=trainable_state(model,head);lora=[p for p in model.parameters() if p.requires_grad]
    # Both arms get an identical fixed step budget, determined BEFORE running A.
    # Deadline interruption marks comparison incomplete, never a result.
    schedule=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(42+epoch);n=rng.permutation(len(normals));a=rng.permutation(len(anomalies))
        pairs=max(len(n),len(a));micro=[(normals[n[j%len(n)]],anomalies[a[j%len(a)]]) for j in range(pairs)]
        schedule += [(epoch,micro[j:j+4]) for j in range(0,len(micro),4)]
    reports={}
    for arm in ['A','B']:
        restore(model,head,base);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        opt=torch.optim.AdamW([{'params':lora,'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
        best=-float('inf');rows=[];folder=out/arm;folder.mkdir(exist_ok=True)
        for step,(epoch,micros) in enumerate(schedule):
            if time.monotonic()>deadline-120:raise TimeoutError('Formal training budget: incomplete paired comparison')
            model.train();head.train();opt.zero_grad(set_to_none=True);stats=[]
            if step+1<len(schedule) and hasattr(provider,'prefetch'):
                next_epoch,next_micros=schedule[step+1]
                provider.prefetch([r for pair in next_micros for r in pair],next_epoch)
            windows=[provider(r,epoch,False) for pair in micros for r in pair]
            # Four balanced micro-losses averaged == one loss over eight bags:
            # both MIL and SS retain precisely the same denominators.
            labels=[0,1]*len(micros)
            _,info=two_pass(forward,windows,labels,arm=='B',chunk=backward_batch,
                            score_batch=score_batch,pad_batches=True);stats.append(info)
            torch.nn.utils.clip_grad_norm_(lora+list(head.parameters()),1.);opt.step()
            rows.append({'step':step+1,'epoch':epoch,'stats':stats})
            if step+1==len(schedule) or schedule[step+1][0]!=epoch:
                model.eval();head.eval();ap=validate(forward,val,provider)
                rows[-1]['val_video_ap']=ap
                if ap>best:  # Exact tie chooses earlier checkpoint.
                    best=ap;model.save_pretrained(folder/'best_adapter');torch.save(head.state_dict(),folder/'best_head.pt')
                    torch.save(trainable_state(model,head),folder/'best_state.pt')
                    (folder/'selection.json').write_text(json.dumps({'step':step+1,'epoch':epoch,'video_ap':ap,'test_used':False}))
                torch.save({'trainable':trainable_state(model,head),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'step':step+1},folder/'resume.pt')
                (folder/'training.json').write_text(json.dumps(rows))
        reports[arm]={'steps':len(rows),'best_val_video_ap':best}
    return reports

def calibrate_videos(forward,records,provider):
    """Temperature and video F1 threshold are fitted ONLY on calibration videos."""
    records=[r for r in records if r['split']=='cal'];assert records
    zs=[];ys=[]
    for r in records:
        _,z=video_probability(forward,provider(r,0,True));zs.append(z);ys.append(r['label'])
    log_t=torch.tensor(0.,requires_grad=True);labels=torch.tensor(ys,dtype=torch.float32)
    opt=torch.optim.LBFGS([log_t],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();temperature=log_t.clamp(-4,4).exp()
        qs=torch.stack([(z/temperature).sigmoid().topk(max(1,math.ceil(len(z)/8))).values.mean() for z in zs])
        obj=torch.nn.functional.binary_cross_entropy(qs.clamp(1e-7,1-1e-7),labels);obj.backward();return obj
    opt.step(closure);t=float(log_t.detach().clamp(-4,4).exp())
    qs=np.array([float((z/t).sigmoid().topk(max(1,math.ceil(len(z)/8))).values.mean()) for z in zs]);ys=np.array(ys)
    best=(-1.,.5)
    for threshold in np.unique(np.r_[0.,qs,1.]):
        prediction=qs>=threshold;tp=sum(prediction&(ys==1));fp=sum(prediction&(ys==0));fn=sum(~prediction&(ys==1));f1=2*tp/max(1,2*tp+fp+fn)
        if f1>best[0]:best=(f1,float(threshold))
    return {'temperature':t,'video_threshold':best[1],'calibration_video_f1':best[0],'scope':'video only; not an optimal frame threshold','test_used':False}

def calibrate_and_test(rt,manifest,provider,training_out,test_root,out):
    """Restore selected A/B, calibrate independently, then read test annotations.

    Call only after both equal-step formal runs have finished. Training outputs
    cannot be replaced with smoke checkpoints: selection.json is mandatory.
    """
    import hashlib
    from .evaluation import freeze_test_lock,run_frozen_paired
    training_out=Path(training_out);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoints={};calibration={};steps=[]
    for arm in ['A','B']:
        folder=training_out/arm
        rows=json.loads((folder/'training.json').read_text());steps.append(len(rows))
        assert rows and 'val_video_ap' in rows[-1], 'Formal arm did not finish an epoch'
        selection=json.loads((folder/'selection.json').read_text());assert selection['test_used'] is False
        path=folder/'best_state.pt';checkpoints[arm]=str(path)
        restore(rt.model,rt.head,torch.load(path,map_location='cpu',weights_only=True))
        rt.model.eval();rt.head.eval()
        calibration[arm]=calibrate_videos(rt.forward,manifest['records'],provider)
    assert steps[0]==steps[1], 'Incomplete paired training'
    hashes={a:hashlib.sha256(Path(p).read_bytes()).hexdigest() for a,p in checkpoints.items()}
    lock=out/'selection_lock.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        assert previous['checkpoints']==hashes and previous['calibration']==calibration
    else:freeze_test_lock(hashes,calibration,lock)
    # Temporal test labels are read only AFTER checkpoint/calibration locking.
    records=json.loads((Path(test_root)/'manifest.json').read_text())['videos']
    expected={r['name']:r['sha256'] for r in manifest['test_identity']}
    assert {r['name']:r['sha256'] for r in records}==expected
    return run_frozen_paired(rt,records,checkpoints,calibration,test_root,out)
