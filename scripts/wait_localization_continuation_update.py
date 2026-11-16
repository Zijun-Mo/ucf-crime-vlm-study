"""Block without model calls until a durable, changed continuation step exists."""
import argparse,datetime,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.followup_budget import spending
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--control',default='research/localization_v4_continuation/control.json')
parser.add_argument('--seconds',type=int,default=3600)
parser.add_argument('--handled-event',type=Path,help='Mark this recovery event handled only after a verified changed durable update')
args=parser.parse_args()
folder=(ROOT/args.control).parent
control=json.loads((folder/'control.json').read_text());deadline=time.monotonic()+args.seconds
while time.monotonic()<deadline:
    launch=json.loads((folder/'launch.json').read_text());receipt_path=ROOT/launch['receipt']
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('finished_utc'):
        print(json.dumps(dict(event='controller_settled_before_update_audit',receipt=launch['receipt'],status=receipt['status'],error=receipt.get('error')),ensure_ascii=False),flush=True)
        raise SystemExit(1)
    health_path=folder/'monitor/health.json'
    health=json.loads(health_path.read_text()) if health_path.exists() else {}
    progress=health.get('progress',{})
    if health.get('receipt')==launch['receipt'] and progress.get('steps',0)>control['resume_steps']:
        interpreter=Path(subprocess.check_output(['which','colab'],text=True).strip()).resolve().with_name('python')
        files=['/content/ucf_localization_train/continuation_restore_audit.json','/content/ucf_localization_train/continuation_first_optimizer_update.json']
        result=subprocess.run([str(interpreter),str(ROOT/'scripts/read_colab_remote.py'),receipt['session'],*files,'--endpoint',receipt['endpoint'],'--tail','20000'],cwd=ROOT,capture_output=True,text=True,timeout=50)
        if result.returncode==0:
            rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if len(rows)==2 and all(r.get('http_status')==200 for r in rows):
                restored,update=[json.loads(r['text']) for r in rows]
                if update.get('continuation_id')!=control['id']:
                    time.sleep(5);continue # copied historical audit is not evidence of this restart
                assert restored['passed'] and restored['optimizer_restored'] and restored['rng_restored']
                assert update['actual_parameters_changed'] and update['durable_checkpoint_verified']
                assert update['steps']>update['previous_steps']>=control['resume_steps']
                assert update['continuation_id']==control['id']
                assert math.isfinite(update['grad_norm']) and update['grad_norm']>0
                assert update['updated_tensor_sha256']!=update['initial_tensor_sha256']
                assert all(n==update['steps'] for n in update['optimizer_steps'])
                for name,value in [('restore_audit.json',restored),('first_optimizer_update.json',update)]:
                    (folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
                launch.update(status='training',training_started=True,actual_new_update_verified=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_checkpoint_steps=update['steps'],first_update_audit=str((folder/'first_optimizer_update.json').relative_to(ROOT)),budget=spending(ROOT))
                (folder/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2)+'\n')
                if args.handled_event:
                    event_path=ROOT/args.handled_event;event=json.loads(event_path.read_text())
                    event.update(handled_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        action_summary=f"CPU cache manifests and remote archives verified;CPU released;resumed same optimizer/RNG trajectory on one L4;actual durable parameter change verified from{update['previous_steps']}to{update['steps']};no change to loss,sampling or validation;continue event-only monitoring",
                        actual_update_audit=str((folder/'first_optimizer_update.json').relative_to(ROOT)))
                    event_path.write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n')
                print('VERIFIED_NEW_PARAMETER_UPDATE '+json.dumps(update,ensure_ascii=False),flush=True)
                raise SystemExit(0)
    time.sleep(5)
raise SystemExit('No update audit within one hour; inspect existing assignment before any retry')
