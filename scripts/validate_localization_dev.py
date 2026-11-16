"""Verify reviewed visual annotations against the authoritative current split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.localization_training import validate_annotations

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations',type=Path)
    args=parser.parse_args()
    manifest_path=ROOT/'results/learning/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    policy=json.loads((ROOT/'research/localization_v1/policy.json').read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==policy['train_manifest_sha256']
    metadata=json.loads((ROOT/'research/formal_training_metadata.json').read_text())['records']
    data=json.loads(args.annotations.read_text())
    rows=validate_annotations(data,manifest,metadata)
    evidence={}
    for row in data['records']:
        assert row.get('reviewer') and row.get('observable_event_note') and row.get('evidence')
        for name in row['evidence']:
            path=ROOT/name
            assert path.is_file()
            evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(status='passed',videos=len(rows),
        annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        authoritative_manifest_sha256=policy['train_manifest_sha256'],
        temporal_training_supervision=False,test_used=False,evidence=evidence)
    output=args.annotations.with_name('dev_annotation_validation.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'},indent=2))

if __name__=='__main__':main()
