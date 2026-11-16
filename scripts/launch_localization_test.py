"""One bounded, serial L4 inference experiment; persist then release."""
import argparse,configparser,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize,configure_drive_environment
from openjev_dss.colab_control import marked_cell
parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['localization_v2_test','localization_v4_test','localization_v4_epoch5_test','localization_v4_plateau_test'],default='localization_v2_test');args=parser.parse_args()
v4=args.experiment.startswith('localization_v4')
formal=False
experiment_root=ROOT/'research'/args.experiment
policy_relative='research/'+experiment_root.name+'/policy.json'
worker_name='run_localization_test.py'
experiment_budget=3.
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=experiment_root/stamp;folder.mkdir(parents=True)
session='ucf-'+args.experiment.replace('_','-');interpreter=Path(shutil.which('colab')).resolve().with_name('python')
receipt=dict(status='preparing',session=session,gpu='L4',started_utc=stamp,controller_pid=os.getpid(),
    hard_stop_minutes=300 if formal else 90,max_allocation_cu=experiment_budget,formal_test=True,training=False,optimizer_updates=0,experiment=args.experiment)
secret=None
def save():
    p=folder/'receipt.tmp';p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');p.replace(folder/'receipt.json')
def run(command,timeout=90,check=True):
    try:p=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired as e:
        with (folder/'controller.log').open('a') as f:f.write('\nTIMEOUT '+str(command[:4])+'\n')
        raise
    with (folder/'controller.log').open('a') as f:f.write('\nCOMMAND '+str(command[:4])+'\n'+sanitize(p.stdout+p.stderr)+'\n')
    p.stdout=sanitize(p.stdout);p.stderr=sanitize(p.stderr)
    if check and p.returncode:raise RuntimeError('Command failed '+str(command[:3])+'; inspect sanitized controller log')
    return p
def cli(*args,**kwargs):return run(['colab',*args],**kwargs)
def remote_json(path):
    p=run([interpreter,'scripts/read_colab_remote.py',session,path,'--tail','2000000' if formal else '200000'],timeout=55,check=False)
    for line in p.stdout.splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if r.get('http_status')==200:return json.loads(r['text'])
    return None
def execute(source,timeout):
    if isinstance(source,Path):text=source.read_text();name=str(source)
    else:text=source;name='storyboard_control.py'
    if v4:
        cli('sessions',timeout=45)
        target=ROOT/'.colab-work'/('v4_test_control_'+hashlib.sha256(text.encode()).hexdigest()[:16]+'.py');target.write_text(text)
        run([interpreter,'scripts/execute_colab_control.py',session,receipt['endpoint'],target,'--timeout',str(timeout)],timeout=timeout+60)
        return
    code,marker,ack=marked_cell(text,name);target=ROOT/'.colab-work'/('storyboard_'+Path(marker).stem+'.py');target.write_text(code)
    for attempt in range(4):
        try:
            cli('sessions',timeout=60)
            p=cli('exec','-s',session,'-f',target,'--timeout',str(timeout),timeout=timeout+35,check=False)
            if ack in p.stdout:return
            if 'Traceback (most recent call last)' in p.stdout and 'Connection' not in p.stdout:
                raise ValueError('Remote setup Python error; inspect controller log')
            probe=run([interpreter,'scripts/read_colab_remote.py',session,marker],timeout=40,check=False)
            if any(json.loads(line).get('text')=='complete' for line in probe.stdout.splitlines() if line.startswith('{')):return
        except (subprocess.TimeoutExpired,RuntimeError):pass
        time.sleep(5*(attempt+1))
    raise RuntimeError('Control cell did not acknowledge completion: '+name)
try:
    save();configure_drive_environment(ROOT)
    budget=json.loads((ROOT/'research/future_work_v1/budget.json').read_text())
    policy=json.loads((ROOT/policy_relative).read_text())
    assert budget['total_limit_cu']==policy['total_research_budget_cu']
    receipt['usage_before']=cli('usage').stdout;save()
    assert re.search(r'Active assignments:\s*0',receipt['usage_before']),'Existing assignment: no new allocation allowed'
    spent=0.
    for old_path in experiment_root.glob('20*/receipt.json'):
        old=json.loads(old_path.read_text())
        if not old.get('allocated_utc'):continue
        assert old.get('finished_utc'),'Previous experiment allocation is not settled'
        old_rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',old['usage_allocated'])[1])
        old_start=datetime.datetime.fromisoformat(old['allocated_utc']);old_end=datetime.datetime.fromisoformat(old.get('release_observed_utc',old['finished_utc']))
        spent+=(old_end-old_start).total_seconds()*old_rate/3600
    from openjev_dss.followup_budget import available,spending
    phase='v3_test' if v4 else 'teacher_test'
    budget_available=min(experiment_budget-spent,available(ROOT,phase))
    receipt['followup_budget']=spending(ROOT)
    if formal:
        earlier=json.loads((ROOT/'research/localization_v2/final_trial_audit.json').read_text())['budget']['total_rate_time_estimate_cu']
        earlier+=json.loads((ROOT/'research/storyboard_context_v1/result_summary.json').read_text())['all_allocation_rate_time_estimate_cu']
        budget_available=min(budget_available,40.-earlier-spent)
        receipt['overall_budget_guard']=dict(total_cu=40.,previous_experiments_estimate_cu=earlier,previous_formal_attempts_estimate_cu=spent,available_cu=budget_available)
    minutes=min(300 if formal else 90,int(budget_available/1.94*60)-2)
    assert minutes>=20,'Insufficient remaining experiment allowance for another allocation'
    receipt.update(prior_allocations_estimate_cu=spent,remaining_experiment_allowance_cu=budget_available,hard_stop_minutes=minutes)
    policy=json.loads((ROOT/policy_relative).read_text())
    if v4:
        from openjev_dss.followup_test import frozen_test_gate
        gate=frozen_test_gate(ROOT,ROOT/policy_relative)
        listing=run(['rclone','lsf',policy['visual_cache_remote'],'--recursive','--files-only','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],timeout=90).stdout.splitlines()
        missing=set(gate['expected_visual_cache_keys'])-set(listing)
        assert not missing,'Missing immutable visual caches; do not allocate GPU: '+str(sorted(missing)[:3])
        gate['all290_visual_caches_present']=True
        (experiment_root/'preallocation_audit.json').write_text(json.dumps(gate,indent=2)+'\n')
        receipt['preallocation_audit']=str((experiment_root/'preallocation_audit.json').relative_to(ROOT))
    # Fixed output_remote permits exact resume of this one selected checkpoint.
    if formal:
        first=policy['records'][0]['name'];expected=policy['metadata'][first]
        health=json.loads(run(['rclone','lsjson',policy['canonical_remote']+'/'+first,'--stat','--hash',
            '--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR','--retries','1'],timeout=60).stdout)
        assert health['Size']==expected['canonical_bytes']
        if health.get('Hashes',{}).get('sha256'):assert health['Hashes']['sha256']==expected['canonical_sha256']
        receipt['native_drive_preflight']=dict(passed=True,first_canonical=first,bytes=health['Size'])
    (folder/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    original=configparser.ConfigParser();original.read(Path.home()/'.config/rclone/rclone.conf')
    config=configparser.ConfigParser();config['gdrive']=dict(original['gdrive']);config['gdrive']['root_folder_id']='YOUR_DRIVE_ROOT_FOLDER_ID'
    secret_dir=Path(tempfile.mkdtemp(prefix='storyboard-secrets-'));secret_dir.chmod(0o700);secret=secret_dir/'rclone.conf'
    with secret.open('w') as f:config.write(f)
    secret.chmod(0o600)
    archive=ROOT/'.colab-work/storyboard_source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'src').rglob('*.py'):z.write(p,p.relative_to(ROOT))
        for name in ['protocol.json','gpu_preflight_result.json','bootstrap_assets.json','formal_input_archive.json',worker_name]:
            z.write(ROOT/'research'/name,'research/'+name)
        z.write(folder/'policy.json',policy_relative)
        if v4:
            z.write(experiment_root/'registration.json',str((experiment_root/'registration.json').relative_to(ROOT)))
            for name in ['launch_localization_test.py','execute_colab_control.py','initialize_colab_kernel.py','inspect_colab_job.py','monitor_localization_test.py','finalize_localization_test.py']:
                z.write(ROOT/'scripts'/name,'scripts/'+name)
        if formal:z.write(experiment_root/'registration.json','research/storyboard_context_formal/registration.json')
    assert archive.stat().st_size<=4*1024**2,'Deployment archive exceeds upload limit;inspect before GPU allocation'
    receipt['source_archive_bytes']=archive.stat().st_size
    receipt['source_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();receipt['output_remote']=policy['output_remote']
    if v4:
        frozen=experiment_root/('source_'+receipt['source_sha256'][:12]+'.zip');shutil.copyfile(archive,frozen)
        receipt['source_archive']=str(frozen.relative_to(ROOT))
    save()
    for attempt in range(2):
        created=cli('new','-s',session,'--gpu','L4',timeout=180,check=False)
        if created.returncode==0:break
        observed=cli('usage').stdout
        assert 'Active assignments: 0' in observed,'Unsettled allocation; do not retry'
        if attempt==1:raise RuntimeError('L4 allocation failed twice without a backend assignment')
        time.sleep(5)
    receipt['allocated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt['usage_allocated']=cli('usage').stdout
    rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_allocated'])[1]);assert rate>0
    if v4:
        minutes=min(minutes,int(budget_available/rate*60)-2)
        assert minutes>=20,'Fresh L4 quote leaves insufficient bounded inference time'
        receipt['hard_stop_minutes']=minutes
    assert rate*minutes/60<=budget_available
    save()
    if v4:
        info=run([interpreter,'-c',"import json;from colab_cli.common import state;s=state.store.get("+repr(session)+");print(json.dumps(dict(endpoint=s.endpoint,kernel_id=s.kernel_id)))"],timeout=30)
        endpoint=json.loads(info.stdout);receipt['endpoint']=endpoint['endpoint'];save()
        initialized=run([interpreter,'scripts/initialize_colab_kernel.py',session,receipt['endpoint']],timeout=90)
        receipt['kernel_initialization']=json.loads(initialized.stdout);save()
    elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(receipt['allocated_utc'])).total_seconds()
    stop_seconds=max(60,int(minutes*60-elapsed))
    watchdog='import os\nos.environ["UCF_HARD_STOP_SECONDS"]='+repr(str(stop_seconds))+'\nos.environ["RCLONE_HEADER"]='+repr(os.environ['RCLONE_HEADER'])+'\nos.environ["UCF_DRIVE_QUOTA_PROJECT"]='+repr(os.environ['UCF_DRIVE_QUOTA_PROJECT'])+'\n'+(ROOT/'research/job_watchdog.py').read_text()
    execute(watchdog,30)
    for src,dst in [(secret,'/content/ucf-rclone.conf'),(archive,'/content/ucf_project.zip')]:
        run([interpreter,'scripts/upload_colab_small.py',session,src,dst],timeout=150)
    execute(ROOT/'research/bootstrap_transfer.py',120);execute(ROOT/'research/setup_runtime.py',300)
    if v4:execute("import os\nos.environ['UCF_FOLLOWUP_POLICY']="+repr(policy_relative)+"\n",20)
    execute(('''import os,sys,subprocess,time,json
from pathlib import Path
os.environ['UCF_DEADLINE']=str(float(json.loads(Path('/content/ucf_job_deadline.json').read_text())['deadline'])-30)
handle=Path('/content/ucf_background_handle.json')
if not handle.exists():
 log=open('/content/ucf_worker.log','w')
 UCF_BACKGROUND_JOB=subprocess.Popen([sys.executable,'/content/ucf_project/research/WORKER_NAME'],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
 ticks=Path('/proc/'+str(UCF_BACKGROUND_JOB.pid)+'/stat').read_text().split(') ',1)[1].split()[19]
 handle.write_text(json.dumps(dict(pid=UCF_BACKGROUND_JOB.pid,start_ticks=ticks,started=time.time(),group='localization_v2_test')))
print('STORYBOARD_WORKER',handle.read_text())
''').replace('WORKER_NAME',worker_name).replace("group='localization_v2_test'","group="+repr(args.experiment)),45)
    receipt['status']='running';save();print('STORYBOARD_RUNNING',folder,flush=True)
    cutoff=time.monotonic()+max(60,minutes*60-(time.time()-datetime.datetime.fromisoformat(receipt['allocated_utc']).timestamp()))+120;polls=0
    while time.monotonic()<cutoff:
        observed=run([interpreter,'scripts/inspect_colab_job.py',session,'--progress','/content/ucf_followup/status.json','--worker'],timeout=65,check=False)
        try:observation=json.loads(observed.stdout)
        except ValueError:observation={}
        receipt['last_observation']=observation;save()
        if observation.get('worker_alive') is False:
            result=remote_json('/content/ucf_followup/result.json');status=remote_json('/content/ucf_followup/status.json')
            # A lost HTTP reply is not a failed inference job. Recover the
            # already-persisted result, without allocating or rescoring.
            if result is None and (status or observation.get('progress',{})).get('status')=='complete':
                recovered=folder/'result_from_drive.json'
                fetched=run(['rclone','copyto',policy['output_remote']+'/result.json',recovered,
                    '--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],timeout=60,check=False)
                if fetched.returncode==0:
                    result=json.loads(recovered.read_text())
                    from openjev_dss.followup_test import result_marker_gate
                    result_marker_gate(policy,result)
                    receipt['result_observation_recovery']=dict(source='persisted Drive result after GPU HTTP timeout',new_gpu_allocations=0,recomputed_predictions=0)
            if result:(folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            if status:(folder/'worker_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
            if result and result['status']=='complete':receipt['status']='complete';break
            raise RuntimeError('Inference worker exited without a complete result: '+str((status or {}).get('error','no status'))[-1800:])
        if observation.get('assignment_present') is False:raise RuntimeError('Backend assignment disappeared; no automatic reallocation')
        polls+=1
        if polls%3==0:
            try:
                cli('sessions',timeout=30,check=False)
                if v4:execute(ROOT/'research/heartbeat_worker.py',10)
                else:cli('exec','-s',session,'-f','research/heartbeat_worker.py','--timeout','10',timeout=25,check=False)
            except Exception:pass
        time.sleep(20)
    else:raise TimeoutError('Bounded inference controller deadline')
except Exception as e:
    receipt.update(status='error',error=sanitize(str(e))[-2500:]);print('STORYBOARD_ERROR',receipt['error'],flush=True)
finally:
    try:
        if receipt.get('allocated_utc'):
            log=run([interpreter,'scripts/read_colab_remote.py',session,'/content/ucf_worker.log','--tail','12000'],timeout=40,check=False)
            (folder/'worker_tail.jsonl').write_text(log.stdout)
            stopped=cli('stop','-s',session,timeout=90,check=False);receipt['stop_exit_code']=stopped.returncode
            receipt['stop_output']=stopped.stdout
        receipt['usage_after']=cli('usage',timeout=40).stdout
        if receipt.get('allocated_utc') and 'Active assignments: 0' in receipt['usage_after']:
            receipt['release_observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    except Exception as e:receipt['cleanup_error']=sanitize(str(e))
    if secret:secret.unlink(missing_ok=True);secret.parent.rmdir()
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print('STORYBOARD_SETTLED',json.dumps(receipt,ensure_ascii=False),flush=True)
