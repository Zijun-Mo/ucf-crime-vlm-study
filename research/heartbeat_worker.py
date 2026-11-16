"""Lightweight kernel activity and actual child-process observation."""
import json
from pathlib import Path

if 'UCF_BACKGROUND_JOB' in globals():
    code = UCF_BACKGROUND_JOB.poll()
    Path('/content/ucf_worker_exit.json').write_text(json.dumps({'returncode': code}) + '\n')
    print('WORKER_RETURNCODE', code)
else:
    print('KERNEL_HEARTBEAT; use the PID/start-time handle for worker liveness')
