"""Idempotently launch one remote process; preserve a PID/start-time handle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

handle_path = Path('/content/ucf_background_handle.json')

def process_stat(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return None
    fields = path.read_text().split(') ', 1)[1].split()
    return {'state': fields[0], 'start_ticks': fields[19]}

if handle_path.exists():
    previous = json.loads(handle_path.read_text())
    actual = process_stat(previous['pid'])
    if actual and actual['start_ticks'] == previous['start_ticks'] and actual['state'] != 'Z':
        print('EXISTING_WORKER', json.dumps(previous))
    else:
        raise RuntimeError('Previous worker is terminal; inspect its results before a new attempt')
else:
    group = os.environ['UCF_GROUP']
    script = ('run_scale_probe.py' if os.environ.get('UCF_SCALE_PROBE') == '1' else
              'run_localization_training.py' if group == 'localization_train' else
              'run_formal_training.py' if group == 'train_formal' else
              'run_cache_prepared_inference.py' if group in ['formal_mil','openjev_zs','qwen_zs','qwen_9b','qwen_27b','qwen_mil'] else
              'run_qwen_mil_training.py' if group == 'train' else 'run_inference_group.py')
    log = open('/content/ucf_worker.log', 'w', buffering=1)
    UCF_BACKGROUND_JOB = subprocess.Popen([sys.executable, '/content/ucf_project/research/' + script],
                                         stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    actual = process_stat(UCF_BACKGROUND_JOB.pid)
    handle = {'pid': UCF_BACKGROUND_JOB.pid, 'start_ticks': actual['start_ticks'], 'group': group,
              'script': script, 'started': time.time()}
    handle_path.write_text(json.dumps(handle, indent=2) + '\n')
    print('STARTED_WORKER', json.dumps(handle))

# Qwen's new duration caches still need canonical videos. Warm at most two
# read-only inputs ahead without changing the scoring worker or its fingerprint.
current=json.loads(handle_path.read_text())
if current['group']=='qwen_zs':
    helper_path=Path('/content/ucf_prefetch_handle_v2.json')
    previous=json.loads(helper_path.read_text()) if helper_path.exists() else None
    actual=process_stat(previous['pid']) if previous else None
    if not (actual and actual['state']!='Z' and actual['start_ticks']==previous['start_ticks']):
        helper_log=open('/content/ucf_prefetch_v2.log','a',buffering=1)
        helper=subprocess.Popen([sys.executable,'/content/ucf_project/research/prefetch_canonical_v2.py'],
                                stdout=helper_log,stderr=subprocess.STDOUT,env=os.environ.copy())
        actual=process_stat(helper.pid)
        helper_path.write_text(json.dumps({'pid':helper.pid,'start_ticks':actual['start_ticks'],
            'worker_pid':current['pid'],'started':time.time()})+'\n')
    print('READ_ONLY_PREFETCH',helper_path.read_text())
