"""Wait locally without model calls; summarize metrics on validation changes."""
import json,subprocess,sys,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];state_path=ROOT/'research/localization_v2/controller_state.json'
last=None
while True:
 try:
  state=json.loads(state_path.read_text());receipt=json.loads((ROOT/state['current_receipt']).read_text())
  progress=receipt.get('last_observation',{}).get('progress',{})
  if progress.get('steps',0)>0 and not (ROOT/'research/localization_v2/first_optimizer_update.json').exists():
   record=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=state['current_receipt'],remote_committed_steps=progress['steps'],stage=progress.get('stage'),video_cursor=progress.get('video_cursor'))
   (ROOT/'research/localization_v2/first_optimizer_update.json').write_text(json.dumps(record,indent=2)+'\n');print('VERIFIED_PARAMETER_UPDATES',record,flush=True)
  marker=(progress.get('epoch_cursor'),len(progress.get('validation',[])))
  terminal=state['status'] in ['complete','budget_complete','needs_inspection','allocation_limit_reached']
  if (progress.get('baseline') and marker!=last) or terminal:
   result=subprocess.run([sys.executable,str(ROOT/'scripts/summarize_localization_trial.py')],cwd=ROOT,capture_output=True,text=True,timeout=140)
   if result.returncode==0:print(result.stdout,flush=True);last=marker
   else:print('METRICS_NOT_READY',result.returncode,flush=True)
  if terminal:break
 except (FileNotFoundError,ValueError,KeyError,subprocess.TimeoutExpired):pass
 time.sleep(20)
