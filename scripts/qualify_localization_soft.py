"""Verify frozen teacher identities, soft targets, labels and budget; no GPU."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.localization_training import validate_annotations,digest
from openjev_dss.localization_soft import targets,window_budget
from openjev_dss.localization_focal_audit import development_metrics
from openjev_dss.followup_budget import available,spending
from openjev_dss.learning import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,default=ROOT/'.colab-work/localization_v3_teacher')
    args=parser.parse_args();folder=ROOT/'research/localization_v4'
    policy_path=folder/'policy.json';policy=json.loads(policy_path.read_text())
    manifest_path=ROOT/'results/learning/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==policy['train_manifest_sha256']
    metadata_path=ROOT/'research/formal_training_metadata.json'
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest()==policy['training_metadata_bundle_sha256']
    metadata=json.loads(metadata_path.read_text())['records']
    annotation_path=ROOT/policy['validation']['annotations'];annotations=json.loads(annotation_path.read_text())
    dev=validate_annotations(annotations,manifest,metadata);assert len(dev)==34
    for r in annotations['records']:
        assert r.get('reviewer') and r.get('evidence') and r.get('observable_event_note')
        assert all((ROOT/e).is_file() for e in r['evidence'])
    records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    old_policy=json.loads((ROOT/'research/localization_v2/policy.json').read_text())
    for k in ['backbone_revision','teacher','input']:assert old_policy[k]==policy[k]
    fingerprint=digest(dict(manifest=manifest,policy=old_policy))
    scores={};hashes={};diagnostics=[]
    for r in records:
        f=args.teacher/(r['name']+'.json');raw=f.read_bytes();entry=json.loads(raw)
        assert entry['fingerprint']==fingerprint and entry['source_sha256']==r['sha256']
        assert entry['teacher']=='pinned_zero_shot_adapter_disabled'
        assert len(entry['logits'])==metadata[r['name']]['window_count']
        scores[r['name']]=entry['logits'];hashes[f.name]=hashlib.sha256(raw).hexdigest()
        if r['split']=='train':
            chosen=targets(entry['logits'],r['label'],policy['loss']['minimum_probability_range'])
            y=np.asarray(chosen['targets']);z=np.asarray(entry['logits'])
            p=torch.sigmoid(torch.tensor(z)).numpy()
            diagnostics.append(dict(name=r['name'],label=r['label'],active=chosen['active'],
                probability_range=chosen['probability_range'],mean_target=float(y.mean()),
                initial_mean_abs_score_gradient=float(np.mean(np.abs(p-y))),windows=len(y)))
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    assert len(train)==1256 and len(val)==163
    budget=spending(ROOT);assert available(ROOT,'v3_training')>0
    baseline=development_metrics(val,scores,dev,metadata);atomic_json(folder/'frame_dev_baseline.json',baseline)
    atomic_json(folder/'target_diagnostics.json',dict(videos=diagnostics,train_only=True,
        abstained_abnormal_videos=sum(r['label'] and not r['active'] for r in diagnostics),
        purity_gate=False,interpretation='Min-max preserves teacher ordering but stretches even weak differences; efficacy requires development evaluation'))
    report=dict(passed=True,status='qualified',gpu_allocated=False,test_used=False,
        policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        teacher_json_sha256=hashes,teacher_videos=len(records),
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/openjev_dss').glob('localization_*.py')},
        baseline=baseline,budget=budget,window_budget=window_budget(records,metadata,policy['sampling']),
        remaining_checks=['actual L4 gradient and <=3h complete-epoch forecast before optimizer steps'])
    atomic_json(folder/'cpu_preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['teacher_json_sha256','sources']},ensure_ascii=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
