"""Stage compatible immutable vision cache files with bounded native transfers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.study_visuals import StudyVisualStore

started=time.time();probe=int(os.environ.get('UCF_FEATURE_PREFETCH_PROBE','0'))
root=Path('/content/ucf_drive');out=Path('/content/ucf_feature_prefetch.json')
local=Path('/content/ucf_feature_bulk_probe') if probe else Path('/content/ucf_visual_cache')
local.mkdir(exist_ok=True)
report={'status':'running','probe':bool(probe),'started':started,'read_only_source':True,'score_files_written':False}
try:
    config=json.loads((root/'formal_stride1/config.json').read_text())
    canonical={r['name']:r for r in config['canonical']}
    records=json.loads((root/'formal_stride1/manifest.json').read_text())['videos']
    store=StudyVisualStore(None,root/'formal_stride1/canonical',canonical,local,'15852e8c16360a2fea060d615a32b45270f8a8fc')
    names=set()
    if probe:
        assert 1<=probe<=4
        records=records[:probe];durations=[4]
    else:durations=[4]  # Shared by all three prompts and both formal MIL arms.
    for r in records:
        m=canonical[r['name']]
        for duration in durations:
            base={**store.signature,'canonical_sha256':m['canonical_sha256'],'source_sha256':r['sha256']}
            variants=[{**base,'representation':'temporal_patch_pairs','pair_batch':4*duration,'duration_gate':duration}]
            if not probe:variants.append({**base,'duration':duration})
            for signature in variants:
                names.add(hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()+'.pt')
    listing=Path('/content/ucf_feature_prefetch_files.txt');listing.write_text('\n'.join(sorted(names))+'\n')
    rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s',
        '--retries','3','--retries-sleep','2s','--log-level','ERROR']
    remote='gdrive:remaining_20261108_v1/visual_features'
    inventory=json.loads(checked(rc+['size',remote,'--files-from',str(listing),'--json'],timeout=90).stdout)
    existing=sum((local/n).stat().st_size for n in names if (local/n).exists())
    assert inventory['bytes']+32*1024**3<shutil.disk_usage(local).free+existing,'Native feature staging exceeds disk reserve'
    report.update(signature=store.signature,inventory=inventory,candidate_names=len(names),
                  candidates_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),existing_local_bytes=existing)
    if inventory['count']:
        timeout=min(900,max(180,120+inventory['bytes']/(8*1024**2)))
        checked(rc+['copy',remote,str(local),'--files-from',str(listing),'--transfers','4','--checkers','4','--inplace=false','--partial-suffix','.ucf-partial'],timeout=timeout)
    staged=[local/n for n in sorted(names) if (local/n).exists()]
    assert len(staged)==inventory['count'] and sum(p.stat().st_size for p in staged)==inventory['bytes']
    if probe:
        assert len(staged)==probe
        comparisons=[]
        for p in staged:
            original=Path('/content/ucf_visual_cache')/p.name
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            assert sha==hashlib.sha256(original.read_bytes()).hexdigest()
            comparisons.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha,'original_cache_bitwise_equal':True})
        report['comparisons']=comparisons
    report.update(status='passed',staged_files=len(staged),staged_bytes=sum(p.stat().st_size for p in staged),
                  note='The unmodified scoring provider still applies its original train-fixture gate and cache-signature checks before use')
except Exception:
    # This is an optional transport preparation. The numerical worker can still
    # read the same immutable entries through the original mount.
    report.update(status='not_ready',error=sanitize(traceback.format_exc()))
report['elapsed_s']=time.time()-started
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
