"""Independent Colab release timer for a full, checkpointed experiment group."""
import json
import os
from pathlib import Path
import threading
import time

UCF_JOB_HARD_DEADLINE = time.time() + int(os.environ['UCF_HARD_STOP_SECONDS'])
Path('/content/ucf_job_deadline.json').write_text(json.dumps({'deadline': UCF_JOB_HARD_DEADLINE}) + '\n')

def _ucf_release_at_deadline():
    time.sleep(max(0, UCF_JOB_HARD_DEADLINE - time.time()))
    try:
        from google.colab import runtime
        runtime.unassign()
    finally:
        os._exit(0)

threading.Thread(target=_ucf_release_at_deadline, daemon=True).start()
print('UCF_JOB_HARD_DEADLINE', UCF_JOB_HARD_DEADLINE)
