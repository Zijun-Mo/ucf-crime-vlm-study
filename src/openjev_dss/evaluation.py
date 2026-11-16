"""Paired test evaluation AFTER freezing A/B and calibration choices."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def aggregate(record,prediction,stride):
    n=record['n_frames'];fps=record['fps'];sums=np.zeros(n);count=np.zeros(n,dtype=np.int32)
    for start,z in zip(prediction['start_s'],prediction['raw_logits']):
        if start%stride:continue
        lo=round(start*fps);hi=min(n,round((start+4)*fps));sums[lo:hi]+=z;count[lo:hi]+=1
    assert np.all(count>0),'Uncovered original frames'
    return sums/count

def metrics(y,z,temperature=1):
    prediction=z>=0  # Fixed 0.5 sigmoid threshold, unchanged by positive temperature.
    positive=y==1;negative=~positive
    return {'auroc':float(roc_auc_score(y,z)),'ap':float(average_precision_score(y,z)),
        'recall_at_0.5':float(prediction[positive].mean()),'fpr_at_0.5':float(prediction[negative].mean()),'temperature':temperature}

def paired_report(records,arms,out,bootstrap=1000,seed=42):
    """arms[A/B]: keyed by video, raw logits for ALL integer starts, same inputs."""
    assert len(records)==290 and set(arms)=={'A','B'}
    for arm in arms:assert set(arms[arm])=={r['name'] for r in records}
    result={'scope':'one frozen paired test, no model or loss selection','strides':{}}
    for stride in [1,2,4]:
        truth=[];scores={arm:[] for arm in arms};abnormal=[]
        for r in records:
            y=np.zeros(r['n_frames'],dtype=np.uint8)
            for a,b in r['intervals']:
                if a>=0 and b>=a:y[max(0,a-1):min(len(y),b)]=1
            truth.append(y);abnormal.append(bool(y.any()))
            for arm in arms:
                p=arms[arm][r['name']];assert p['start_s']==list(range(math.ceil(r['n_frames']/r['fps'])))
                scores[arm].append(aggregate(r,p,stride))
        y=np.concatenate(truth);rows={}
        abnormal_y=np.concatenate([v for v,a in zip(truth,abnormal) if a])
        for arm in arms:
            rows[arm]={'all_frames':metrics(y,np.concatenate(scores[arm])),
                'abnormal_videos':metrics(abnormal_y,np.concatenate([v for v,a in zip(scores[arm],abnormal) if a]))}
        rng=np.random.default_rng(seed);differences={'auroc':[],'ap':[],'recall_at_0.5':[],'fpr_at_0.5':[]};abnormal_differences={k:[] for k in differences}
        for _ in range(bootstrap):
            idx=rng.integers(0,len(records),len(records));by=np.concatenate([truth[i] for i in idx])
            if len(np.unique(by))<2:continue
            bm={a:metrics(by,np.concatenate([scores[a][i] for i in idx])) for a in arms}
            for key in differences:differences[key].append(bm['B'][key]-bm['A'][key])
            ai=[i for i in idx if abnormal[i]]
            if ai:
                ay=np.concatenate([truth[i] for i in ai])
                if len(np.unique(ay))==2:
                    am={a:metrics(ay,np.concatenate([scores[a][i] for i in ai])) for a in arms}
                    for key in abnormal_differences:abnormal_differences[key].append(am['B'][key]-am['A'][key])
        rows['paired_B_minus_A_95pct_CI']={'all_frames':{k:np.quantile(v,[.025,.975]).tolist() for k,v in differences.items()},'abnormal_videos':{k:np.quantile(v,[.025,.975]).tolist() for k,v in abnormal_differences.items() if v}}
        result['strides'][str(stride)]=rows
    Path(out).write_text(json.dumps(result,indent=2));return result

def freeze_test_lock(checkpoints,calibration,out):
    """Persist choice hashes before reading test temporal annotations."""
    choice={'checkpoints':checkpoints,'calibration':calibration}
    choice['sha256']=hashlib.sha256(json.dumps(choice,sort_keys=True).encode()).hexdigest()
    Path(out).write_text(json.dumps(choice,indent=2));return choice

def run_frozen_paired(rt,records,checkpoints,calibration,source_root,out):
    """One locked paired test, with resumable per-video raw-score artifacts."""
    import torch,shutil
    from .runtime import VisualStore
    from .training import restore
    from .experiment import score_windows
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source_root=Path(source_root)
    assert set(checkpoints)==set(calibration)=={'A','B'}
    hashes={a:hashlib.sha256(Path(p).read_bytes()).hexdigest() for a,p in checkpoints.items()}
    lock_path=out/'selection_lock.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text());assert lock['checkpoints']==hashes and lock['calibration']==calibration
    else:lock=freeze_test_lock(hashes,calibration,lock_path)
    assert len(records)==290
    cfg=json.loads((source_root/'config.json').read_text());canon={r['name']:r for r in cfg['canonical']}
    prepared=out/'prepared';(prepared/'canonical').mkdir(parents=True,exist_ok=True);(prepared/'metadata').mkdir(exist_ok=True)
    for r in records:
        m=canon[r['name']];assert m['source_sha256']==r['sha256']
        target=prepared/'canonical'/r['name']
        if not target.exists():shutil.copyfile(source_root/'canonical'/r['name'],target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==m['canonical_sha256']
        metadata={**m,'sha256':r['sha256'],'n_frames':r['n_frames'],'fps':r['fps']}
        (prepared/'metadata'/(r['name']+'.json')).write_text(json.dumps(metadata))
    provider=VisualStore(rt,prepared);arms={}
    for arm,path in checkpoints.items():
        restore(rt.model,rt.head,torch.load(path,map_location='cpu',weights_only=True));rt.model.eval();rt.head.eval();folder=out/arm;folder.mkdir(exist_ok=True);arms[arm]={}
        for r in records:
            target=folder/(r['name']+'.json')
            if target.exists():
                p=json.loads(target.read_text());assert p['selection_sha256']==lock['sha256']
            else:
                windows=provider(r,0,True);raw=score_windows(rt.forward,windows,4).tolist()
                p={'name':r['name'],'start_s':list(range(len(raw))),'raw_logits':raw,'selection_sha256':lock['sha256'],'temperature':calibration[arm]['temperature']}
                target.write_text(json.dumps(p))
            arms[arm][r['name']]=p
    return paired_report(records,arms,out/'paired_test_report.json')
