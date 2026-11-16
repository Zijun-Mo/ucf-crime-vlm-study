"""Block without model calls until the fixed test emits a verifiable prediction."""
import argparse,datetime,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
folder=ROOT/'research'/a.experiment;policy=json.loads((folder/'policy.json').read_text())
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    launch=json.loads((folder/'launch.json').read_text());receipt_path=ROOT/launch['receipt'];receipt=json.loads(receipt_path.read_text());progress=receipt.get('last_observation',{}).get('progress',{})
    if receipt.get('finished_utc') and receipt['status']!='complete':
        print(json.dumps(dict(status='controller_failed',receipt=launch['receipt'],error=receipt.get('error'))),flush=True);raise SystemExit(1)
    if progress.get('completed_videos',0)>0:
        assert progress.get('optimizer_updates')==0
        first=policy['records'][0];base='/content/ucf_followup/'
        files=[base+'predictions/'+first['name']+'.json',base+'checkpoint_audit.json',base+'cache_gate.json']
        interpreter=Path(shutil.which('colab')).resolve().with_name('python')
        r=subprocess.run([str(interpreter),str(ROOT/'scripts/read_colab_remote.py'),receipt['session'],*files,'--endpoint',receipt['endpoint'],'--tail','200000'],cwd=ROOT,capture_output=True,text=True,timeout=55)
        rows=[json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
        if r.returncode==0 and len(rows)==3 and all(x.get('http_status')==200 for x in rows):
            prediction,checkpoint,cache=[json.loads(x['text']) for x in rows]
            assert prediction['fingerprint']==policy['fingerprint'] and prediction['source_sha256']==first['sha256']
            assert prediction['canonical_sha256']==policy['metadata'][first['name']]['canonical_sha256']
            assert prediction['selected_epoch']==checkpoint['selected_epoch']==policy['selected_epoch']
            assert prediction['checkpoint_tensor_sha256']==checkpoint['tensor_sha256']==policy['checkpoint_tensor_sha256']
            assert checkpoint['file_sha256']==policy['checkpoint_file_sha256'] and checkpoint['selection_frozen_before_test']
            assert cache['allowed'] and len(prediction['raw_logits'])==len(prediction['start_s'])==math.ceil(first['n_frames']/first['fps'])
            assert all(math.isfinite(z) for z in prediction['raw_logits'])
            now=datetime.datetime.now(datetime.timezone.utc).isoformat()
            audit=dict(passed=True,verified_utc=now,selected_epoch=policy['selected_epoch'],first_video=first['name'],first_windows=len(prediction['raw_logits']),first_prediction_sha256=hashlib.sha256(rows[0]['text'].encode()).hexdigest(),checkpoint=checkpoint,cache_gate=cache,optimizer_updates=0,completed_videos=progress['completed_videos'],scored_windows=progress.get('scored_windows'),receipt=launch['receipt'])
            for name,value in [('first_prediction_audit.json',audit),('first_prediction.json',prediction)]:
                (folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
            launch.update(status='scoring',test_scoring_verified=True,verified_utc=now,last_observed_completed_videos=progress['completed_videos'],first_prediction_audit=str((folder/'first_prediction_audit.json').relative_to(ROOT)))
            (folder/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2)+'\n')
            reg=json.loads((folder/'registration.json').read_text());reg.update(status='running',receipt=launch['receipt'],test_predictions_produced=True,actual_first_prediction_audit=launch['first_prediction_audit']);(folder/'registration.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
            ident=hashlib.sha256((str(receipt_path.resolve())+':first_prediction').encode()).hexdigest()[:24];event=folder/'monitor/events'/(ident+'.json');event.parent.mkdir(parents=True,exist_ok=True)
            e=json.loads(event.read_text()) if event.exists() else dict(id=ident,kind='first_prediction',receipt=str(receipt_path),created_utc=now)
            e.update(handled_utc=now,action_summary='Actual first prediction, selected checkpoint hashes, cache gate, window coverage and zero optimizer updates verified',audit=launch['first_prediction_audit']);event.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n')
            print('FIRST_PREDICTION_VERIFIED '+json.dumps({k:v for k,v in audit.items() if k not in ['cache_gate','checkpoint']},ensure_ascii=False),flush=True);raise SystemExit(0)
    time.sleep(5)
raise SystemExit('No first prediction within30minutes;inspect existing assignment before any retry')
