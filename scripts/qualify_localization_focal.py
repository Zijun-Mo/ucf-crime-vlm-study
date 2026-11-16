"""No-GPU V3 gate: exact teacher identities, 34 reviewed labels, .80/.90 overlap."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.localization_training import validate_annotations,digest
from openjev_dss.localization_objective import normal_reference
from openjev_dss.localization_focal import select,calibration
from openjev_dss.localization_focal_audit import pseudo_label_audit,development_metrics
from openjev_dss.learning import atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--teacher',type=Path,default=ROOT/'.colab-work/localization_v3_teacher')
    args=parser.parse_args();folder=ROOT/'research/localization_v3';policy_path=folder/'policy.json';policy=json.loads(policy_path.read_text())
    report=dict(passed=False,status='pending',gpu_allocated=False,test_used=False,policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
    path=ROOT/policy['validation']['annotations']
    if not path.exists():
        report.update(reason='Awaiting 12 additional Hydrogen annotations and independent image/boundary review',expected_anomaly_videos=34)
        atomic_json(folder/'cpu_preflight.json',report);print(json.dumps(report));return 2
    manifest=json.loads((ROOT/'results/learning/manifest.json').read_text());metadata=json.loads((ROOT/'research/formal_training_metadata.json').read_text())['records']
    assert hashlib.sha256((ROOT/'results/learning/manifest.json').read_bytes()).hexdigest()==policy['train_manifest_sha256']
    annotations=json.loads(path.read_text());dev=validate_annotations(annotations,manifest,metadata)
    assert len(dev)==34
    old_names={r['name'] for r in json.loads((ROOT/'research/localization_v1/dev_annotations.json').read_text())['records']}
    new_names={r['name'] for r in json.loads((ROOT/'research/localization_dev_v2/candidates.json').read_text())['records']}
    assert set(dev)==old_names|new_names
    for row in annotations['records']:
        assert row.get('reviewer') and row.get('evidence') and row.get('observable_event_note')
        for name in row['evidence']:assert (ROOT/name).is_file()
    source_policy=json.loads((ROOT/'research/localization_v2/policy.json').read_text());fp=digest(dict(manifest=manifest,policy=source_policy));records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];scores={};hashes={}
    for row in records:
        f=args.teacher/(row['name']+'.json');raw=f.read_bytes();entry=json.loads(raw)
        assert entry['fingerprint']==fp and entry['source_sha256']==row['sha256'] and entry['teacher']=='pinned_zero_shot_adapter_disabled'
        assert len(entry['logits'])==metadata[row['name']]['window_count'];scores[row['name']]=entry['logits'];hashes[f.name]=hashlib.sha256(raw).hexdigest()
    reference=normal_reference([scores[r['name']] for r in train if not r['label']],policy['selection']['normal_quantile'])
    selections={r['name']:select(scores[r['name']],reference,policy['selection']) for r in train if r['label']}
    gate=pseudo_label_audit(dev,scores,reference,policy['selection']);atomic_json(folder/'pseudo_label_quality_gate.json',gate)
    positive=calibration(train,scores,selections);atomic_json(folder/'positive_calibration_preflight.json',positive)
    baseline=development_metrics(val,scores,dev,metadata);atomic_json(folder/'frame_dev_baseline.json',baseline)
    report.update(passed=gate['passed'],status='qualified' if gate['passed'] else 'quality_gate_failed',
        reason=None if gate['passed'] else 'Frozen positive/background pseudo-label overlaps fail the preregistered .80/.90 thresholds; no training or speed-probe GPU may be allocated',
        annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),teacher_json_sha256=hashes,
        quality_gate=gate,positive_calibration=positive,baseline=baseline,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/openjev_dss').glob('localization_*.py')})
    atomic_json(folder/'cpu_preflight.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['sources','teacher_json_sha256','quality_gate','baseline']}));return 0 if gate['passed'] else 3
if __name__=='__main__':raise SystemExit(main())
