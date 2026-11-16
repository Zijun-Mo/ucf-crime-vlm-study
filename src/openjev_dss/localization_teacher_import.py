"""Reuse identical native teacher outputs across distinct student protocols."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .learning import atomic_json


def fingerprint(manifest,policy):
    return hashlib.sha256(json.dumps(dict(manifest=manifest,policy=policy),sort_keys=True).encode()).hexdigest()


def import_teacher(source,target,manifest,metadata,old_policy,new_policy):
    for key in ['backbone_revision','train_manifest_sha256','training_metadata_bundle_sha256','teacher']:
        assert old_policy[key]==new_policy[key],f'Teacher identity changed: {key}'
    for key in ['prompt','duration','stride','batch']:
        assert old_policy['input'][key]==new_policy['input'][key],f'Teacher input changed: {key}'
    old_fp=fingerprint(manifest,old_policy);new_fp=fingerprint(manifest,new_policy)
    records={r['name']:r for r in manifest['records'] if r['split'] in ['train','val']}
    target=Path(target);target.mkdir(exist_ok=True,parents=True);imported=[]
    for path in sorted(Path(source).glob('*.json')):
        data=json.loads(path.read_text());record=records[data['name']]
        assert data['fingerprint']==old_fp and data['source_sha256']==record['sha256']
        assert data['windows']==metadata[data['name']]['window_count']==len(data['logits'])
        assert np.isfinite(data['logits']).all() and data['teacher']=='pinned_zero_shot_adapter_disabled'
        dest=target/path.name
        if dest.exists():continue
        atomic_json(dest,{**data,'fingerprint':new_fp,'imported_from':dict(protocol='localization_v1',
            fingerprint=old_fp,source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),values_unchanged=True)})
        imported.append(data['name'])
    return dict(imported_videos=len(imported),names=imported,source_fingerprint=old_fp,
        target_fingerprint=new_fp,teacher_identity_verified=True)
