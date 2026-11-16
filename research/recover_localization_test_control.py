"""Idempotent same-VM recovery when a Colab cell acknowledgement is lost."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/content/ucf_project')
native=json.loads((root/'research/native_drive_policy.json').read_text()) if (root/'research/native_drive_policy.json').exists() else None
# The task quota project is public metadata, not an authentication credential.
if native:
    os.environ['RCLONE_HEADER']='X-Goog-User-Project: '+native['quota_project']
    os.environ['UCF_DRIVE_QUOTA_PROJECT']=native['quota_project']
os.environ['UCF_DEADLINE']=str(json.loads(Path('/content/ucf_job_deadline.json').read_text())['deadline']-30)
handle=Path('/content/ucf_background_handle.json')
if not handle.exists():
    log=open('/content/ucf_worker.log','w')
    worker=subprocess.Popen([sys.executable,str(root/'research/run_localization_test.py')],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    ticks=Path('/proc/'+str(worker.pid)+'/stat').read_text().split(') ',1)[1].split()[19]
    handle.write_text(json.dumps(dict(pid=worker.pid,start_ticks=ticks,started=time.time(),group='localization_v2_test')))
# Mark exactly the stalled idempotent launch cell, never a setup or training cell.
Path('/content/ucf_control_472c6b8f06a246cf953ef63eab15b34e.done').write_text('complete')
print('EXISTING_ASSIGNMENT_WORKER',handle.read_text(),flush=True)
