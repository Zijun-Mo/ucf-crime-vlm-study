"""Require a complete frozen native teacher cache before GPU training work."""
import json,math
from pathlib import Path


def require_complete_teacher(records,folder,metadata,fingerprint):
    folder=Path(folder);records=[r for r in records if r['split'] in ['train','val']]
    missing=[r['name'] for r in records if not (folder/(r['name']+'.json')).is_file()]
    if missing:raise ValueError(f'Frozen teacher cache incomplete: {len(missing)}/{len(records)} missing; GPU teacher recomputation forbidden; first={missing[:3]}')
    for r in records:
        d=json.loads((folder/(r['name']+'.json')).read_text())
        if (d['source_sha256']!=r['sha256'] or d['fingerprint']!=fingerprint or
            d['teacher']!='pinned_zero_shot_adapter_disabled' or
            d['windows']!=metadata[r['name']]['window_count'] or len(d['logits'])!=d['windows'] or
            not all(math.isfinite(z) for z in d['logits'])):
            raise ValueError('Frozen teacher cache identity/content mismatch: '+r['name'])
    return dict(passed=True,videos=len(records),complete=True,gpu_teacher_recomputation_allowed=False)
