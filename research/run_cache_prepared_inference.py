"""Pipeline optional cache I/O around the unchanged numerical producer."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

group=os.environ['UCF_GROUP'];assert group in ['formal_mil','openjev_zs','qwen_zs','qwen_9b','qwen_27b','qwen_mil']
root=Path('/content/ucf_project/research')
if group in ['formal_mil','openjev_zs']:
    Path('/content/ucf_group_status.json').write_text(json.dumps({'status':'running','stage':'native_feature_prefetch',
        'group':group,'completed':[],'started':time.time()})+'\n')
    try:
        subprocess.run([sys.executable,str(root/'prepare_inference_cache.py')],check=False,timeout=1110)
    except Exception as error:
        print('OPTIONAL_CACHE_PREPARATION',type(error).__name__,flush=True)
# Original providers apply their own gate and signature checks. Atomic native
# downloads leave incomplete files with .ucf-partial, never as usable .pt files.
script='run_formal_inference.py' if group=='formal_mil' else 'run_inference_group.py'
sys.path[:0]=['/content/ucf_deps',str(root.parent/'src')]
import hashlib
import openjev_dss.study_pair_visuals as pair_module
from openjev_dss.optional_cache_sync import OptionalCacheSync
command=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s',
         '--timeout','60s','--retries','2','--low-level-retries','2','--retries-sleep','2s',
         '--stats','30s','--stats-one-line','--log-level','INFO']
uploads=OptionalCacheSync(command,'gdrive:remaining_20261108_v1/visual_features',
                          '/content/ucf_optional_cache_sync.json',float(os.environ['UCF_DEADLINE']))
original=pair_module.GatedPairVisualStore
class PipelinedCacheStore(original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'sync' in kwargs:self.sync=uploads.submit
assert PipelinedCacheStore.get is original.get and PipelinedCacheStore.gate is original.gate
pair_module.GatedPairVisualStore=PipelinedCacheStore
proof={'scope':'Only the cache publication callback changes; original gate/get/scoring methods remain in use',
       'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'original_feature_get_and_gate_methods_unchanged':True,
       'numerical_source_sha256':{name:hashlib.sha256((root.parent/name).read_bytes()).hexdigest() for name in [
           'src/openjev_dss/controlled_runtime.py','src/openjev_dss/study_visuals.py','src/openjev_dss/study_pair_visuals.py',
           'src/openjev_dss/experiment.py','research/run_inference_group.py']}}
Path('/content/ucf_optional_cache_policy.json').write_text(json.dumps(proof,indent=2)+'\n')
try:runpy.run_path(str(root/script),run_name='__main__')
finally:
    uploads.close()
    pair_module.GatedPairVisualStore=original
