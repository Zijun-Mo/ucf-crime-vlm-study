"""Fetch small trial metrics only; never download video, features or weights."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import configure_drive_environment,sanitize
from openjev_dss.learning import atomic_json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',default='research/localization_v2/policy.json');args=p.parse_args()
policy_path=ROOT/args.policy;policy=json.loads(policy_path.read_text());folder=policy_path.parent
configure_drive_environment(ROOT);remote=policy['resources']['output_remote']
base=['rclone','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR','--retries','2']
def read(name,required=True):
 r=subprocess.run(base+['cat',remote+'/'+name],capture_output=True,text=True,timeout=90)
 if r.returncode:
  if required:raise RuntimeError('Cannot read trial metric '+name+': '+sanitize(r.stderr)[-500:])
  return None
 return json.loads(r.stdout)
status=read('status.json');status_source='drive_published_status'
# The worker writes local validation before publishing its checkpoint. A live
# observation can therefore be newer than Drive; do not report an old epoch.
health_path=folder/'monitor/health.json'
if health_path.exists():
 health=json.loads(health_path.read_text());live=health.get('progress',{})
 receipt_path=ROOT/health.get('receipt','missing')
 if receipt_path.is_file() and json.loads(receipt_path.read_text()).get('localization_policy')==args.policy:
  def revision(d):return (len(d.get('validation',[])),d.get('steps',0))
  if revision(live)>revision(status):
   status=live;status_source='monitor_remote_observation_publication_not_confirmed'
baseline=status.get('baseline') or read('baseline_validation.json',False);epochs=[]
for row in status.get('validation',[]):
 result={k:v for k,v in row.items() if k!='sampling_policy'}
 if baseline:
  for key in ['within_macro_auroc','video_max_ap','video_max_auroc']:
   if result.get(key) is not None and baseline.get(key) is not None:result[key+'_change_pp']=100*(result[key]-baseline[key])
 epochs.append(result)
summary=dict(status=status['status'],status_source=status_source,stage=status.get('stage'),steps=status.get('steps'),
 teacher_videos=status.get('teacher_videos'),baseline=baseline,epochs=epochs,best_epoch=status.get('best_epoch'),
 selection_status=status.get('selection_status'),selected_artifact=status.get('selected_artifact'),
 annotations='22 assistant visually reviewed development videos; approximate intervals',test_used=False,
 note='Development comparison; no claim of test-set improvement. Epoch zero can remain selected.')
atomic_json(folder/'result_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
