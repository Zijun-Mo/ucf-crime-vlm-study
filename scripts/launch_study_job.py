"""Run one full Colab study group, preserve results, and always release its VM."""
import argparse
import configparser
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.cloud_io import sanitize, configure_drive_environment
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job', choices=['qwen_zs', 'openjev_zs', 'qwen_mil', 'qwen_9b', 'qwen_27b', 'train', 'train_formal', 'formal_mil', 'localization_train'])
parser.add_argument('--gpu', default='L4', choices=['L4', 'G4', 'A100', 'H100'])
parser.add_argument('--minutes', type=int, default=180)
parser.add_argument('--probe', action='store_true', help='Training-fixture-only scale probe; no test scoring')
parser.add_argument('--epoch-limit', type=int, default=10, help='Formal training epoch budget; a limit is not convergence')
parser.add_argument('--localization-phase', choices=['prepare','train','auto'],default='auto',help='Auto prepares teacher, then trains only with reviewed temporal dev labels')
parser.add_argument('--localization-policy',default='research/localization_v1/policy.json',choices=['research/localization_v1/policy.json','research/localization_v2/policy.json','research/localization_v3/policy.json','research/localization_v4/policy.json'])
parser.add_argument('--adopt', type=Path, help='Take over monitoring/settlement/release of an already running worker from this receipt; no allocation or setup')
parser.add_argument('--continuation',help='Authorized execution-only extension; frozen policy stays unchanged')
args = parser.parse_args()
if args.adopt:
    args.localization_policy=json.loads(args.adopt.read_text()).get('localization_policy',args.localization_policy)
    args.continuation=json.loads(args.adopt.read_text()).get('continuation_control',args.continuation)
selected_localization_policy=json.loads((ROOT/args.localization_policy).read_text()) if args.job=='localization_train' else None
from openjev_dss.continuation_control import load_continuation
continuation=load_continuation(ROOT,args.continuation,args.localization_policy) if args.job=='localization_train' else None
assert not args.continuation or continuation
if args.job=='localization_train':
    assert args.gpu=='L4','User prohibits A100 for localization training because of cost'
    if args.localization_policy in ['research/localization_v3/policy.json','research/localization_v4/policy.json']:
        approved=json.loads((ROOT/args.localization_policy).with_name('cpu_preflight.json').read_text())
        assert approved['passed'] and approved['status']=='qualified','Selected protocol CPU / reviewed annotation gate has not passed'
        assert approved['policy_sha256']==hashlib.sha256((ROOT/args.localization_policy).read_bytes()).hexdigest()
        assert approved['annotation_sha256']==hashlib.sha256((ROOT/selected_localization_policy['validation']['annotations']).read_bytes()).hexdigest()
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in approved['sources'].items()),'Localization objective changed after CPU gate'
    if selected_localization_policy['resources'].get('require_pointwise_epoch_profile'):
        smoke_path=(ROOT/args.localization_policy).with_name('publication_live_test.json')
        smoke=json.loads(smoke_path.read_text())
        assert smoke['passed'] and smoke.get('tested_sources'),'Run real publication smoke test before allocating a pointwise trial'
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in smoke['tested_sources'].items()),'Publication code changed after live smoke test'
if args.job=='localization_train' and args.localization_phase!='prepare' and not selected_localization_policy['resources'].get('require_pointwise_epoch_profile'):
    hold=ROOT/'research/localization_v1/optimization_hold.json'
    if hold.exists():
        qualification=ROOT/'research/localization_v1/optimization_qualification.json'
        assert qualification.exists(),'Full localization training held: user requires qualified <=3-hour epochs before launch'
        qualified=json.loads(qualification.read_text())
        assert qualified.get('qualified') is True and qualified['epoch_upper_bound_hours']<=3,'Localization epoch speed requirement has not passed'
adopting = args.adopt is not None
if adopting:
    adopted_receipt = json.loads(args.adopt.read_text())
    assert adopted_receipt['status'] == 'running' and adopted_receipt['job'] == args.job and 'finished_utc' not in adopted_receipt
    args.gpu, args.minutes, args.probe = adopted_receipt['gpu'], adopted_receipt['hard_stop_minutes'], adopted_receipt['probe']
    args.epoch_limit = adopted_receipt.get('epoch_limit', args.epoch_limit)
    args.localization_phase = adopted_receipt.get('localization_phase', args.localization_phase)
if args.job == 'train_formal':
    policy = json.loads((ROOT / 'research/training_protocol_v2.json').read_text())
    assert args.epoch_limit >= policy['epoch_budget']['initial']
    assert (args.epoch_limit-policy['epoch_budget']['initial']) % policy['epoch_budget']['extension_block'] == 0
assert not args.probe or args.job in ['qwen_9b', 'qwen_27b']
assert json.loads((ROOT / 'research/gpu_preflight_result.json').read_text())['status'] == 'passed'
session = 'ucf-' + args.job.replace('_', '-') + ('-probe' if args.probe else '')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
directory = ROOT / 'research/runs' / (args.job + ('_probe' if args.probe else '') + '_' + stamp)
if adopting:
    directory = args.adopt.resolve().parent
    session = adopted_receipt['session']
else:
    directory.mkdir(parents=True)
log_path = directory / 'controller.log'
receipt_path = directory / 'receipt.json'
receipt = {'job': args.job, 'session': session, 'gpu': args.gpu, 'hard_stop_minutes': args.minutes,
           'started_utc': stamp, 'controller_pid': os.getpid(), 'status': 'preparing', 'completed': [], 'probe': args.probe}
if adopting:
    receipt = adopted_receipt
    receipt.setdefault('controller_adoptions', []).append({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'previous_controller_pid': receipt['controller_pid'], 'reason': 'Original controller process exited while the remote worker kept running'})
    receipt['controller_pid'] = os.getpid()
if args.job == 'train_formal':receipt['epoch_limit'] = args.epoch_limit
if args.job == 'localization_train':
    receipt['localization_phase'] = args.localization_phase
    receipt['localization_policy'] = args.localization_policy
    if continuation:
        receipt.update(continuation_control=args.continuation,continuation_id=continuation['id'],
            event_scope=continuation['id'],resume_steps=continuation['resume_steps'],
            validation_epoch_floor=continuation['completed_epochs'],output_remote=continuation['output_remote'],
            startup_without_update_limit_minutes=45)
    receipt['qualification_mode']='mandatory_remote_pointwise_pre_optimizer_profile' if selected_localization_policy['resources'].get('require_pointwise_epoch_profile') else 'local_qualification_required'
    if args.localization_policy=='research/localization_v4/policy.json':
        receipt['startup_without_update_limit_minutes']=45


def clean(text):
    text = re.sub(r'(colab-runtime-proxy-token=)[^\s&]+', r'\1<redacted>', text)
    text = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '<redacted-jwt>', text)
    return sanitize(text)


def save():
    temporary = receipt_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(receipt_path)


def run(command, timeout=180, check=True):
    with log_path.open('a') as log:
        log.write('\nCOMMAND ' + ' '.join(map(str, command)) + '\n')
        log.flush()
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            partial=error.output or ''
            if isinstance(partial,bytes):partial=partial.decode(errors='replace')
            log.write(clean(partial));log.write(f'\nCOMMAND_TIMEOUT {timeout}s\n')
            raise
        result.stdout = clean(result.stdout)
        log.write(result.stdout)
    if check and result.returncode:
        raise RuntimeError(f'Command failed: {command[:3]}, exit={result.returncode}')
    return result


def cli(*command, **kwargs):
    return run(['colab', *command], **kwargs)


def execute(script, timeout):
    localization=args.job=='localization_train'
    if localization and args.localization_policy=='research/localization_v4/policy.json':
        # One submission per cell, fresh HTTP completion evidence. The CLI's
        # extra pre-execution websocket command can lose its connection before
        # transmitting the actual cell; do not spend repeated setup time on it.
        assert receipt.get('endpoint'),'Existing allocation endpoint required'
        interpreter=Path(shutil.which('colab')).resolve().with_name('python')
        return run([str(interpreter),'scripts/execute_colab_control.py',session,
            receipt['endpoint'],str(script),'--timeout',str(timeout)],timeout=timeout+60)
    attempts=4 if localization else 1
    submitted_script=script
    if localization:
        from openjev_dss.colab_control import marked_cell
        cell,marker,acknowledgement=marked_cell(Path(script).read_text(),script)
        submitted_script=ROOT/'.colab-work'/('control_'+Path(marker).stem+'.py')
        submitted_script.write_text(cell)
    for attempt in range(attempts):
        try:
            if localization:
                # Official CLI sync refreshes the proxy even before token expiry.
                cli('sessions',timeout=60)
            result=cli('exec', '-s', session, '-f', str(submitted_script), '--timeout', str(timeout), timeout=timeout + 40)
            if localization and acknowledgement not in result.stdout:
                raise RuntimeError('Remote Python script failed: '+str(script)+'; inspect controller log')
            return result
        except (RuntimeError,subprocess.TimeoutExpired):
            if localization:
                # A lost websocket acknowledgement does not imply a failed cell.
                interpreter=Path(shutil.which('colab')).resolve().with_name('python')
                try:
                    observed=run([str(interpreter),'scripts/read_colab_remote.py',session,marker],timeout=40,check=False)
                except subprocess.TimeoutExpired:
                    observed=subprocess.CompletedProcess([],1,stdout='')
                for line in observed.stdout.splitlines():
                    try:report=json.loads(line)
                    except ValueError:continue
                    if report.get('http_status')==200 and report.get('text')=='complete':
                        return observed
            if attempt+1==attempts:raise
            time.sleep(min(20,5*(attempt+1)))



def observe_worker(progress):
    interpreter = Path(shutil.which('colab')).resolve().with_name('python')
    command = [str(interpreter), 'scripts/inspect_colab_job.py', session, '--progress', progress, '--worker']
    if receipt.get('endpoint'):
        command += ['--endpoint', receipt['endpoint']]
    observed = run(command, timeout=90)
    return json.loads(observed.stdout)


try:
    save()
    if not adopting:
        receipt['usage_before'] = cli('usage').stdout
        assignments=re.search(r'Active assignments:\s*(\d+)',receipt['usage_before'])
        existing_cpu_ok=False
        if args.job=='localization_train' and assignments and int(assignments.group(1))==1:
            inventory=cli('sessions').stdout
            rows=[line for line in inventory.splitlines() if 'Hardware:' in line]
            existing_rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_before'])
            existing_cpu_ok=(len(rows)==1 and '[ucf-localization-dev]' in rows[0] and 'Hardware: CPU' in rows[0]
                and existing_rate is not None and float(existing_rate.group(1))<=.10)
            if existing_cpu_ok:receipt['concurrent_annotation_cpu']={'session':rows[0],'observed_rate_cu_hr':float(existing_rate.group(1))}
        assert assignments and (int(assignments.group(1))==0 or existing_cpu_ok),'An existing assignment must be inspected before allocating another GPU'
        balance_before=re.search(r'Current balance:\s*([0-9.]+)',receipt['usage_before'])
        assert balance_before,'Missing actual balance; do not allocate against assumed credits'
        observed_rates=[]
        for old_path in sorted((ROOT/'research/runs').glob('*/receipt.json')):
            old=json.loads(old_path.read_text())
            if old.get('gpu')!=args.gpu or not old.get('allocated_utc'):
                continue
            rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',old.get('usage_allocated',old.get('usage_at_allocation','')))
            if rate and float(rate.group(1))>0:
                observed_rates.append((old['allocated_utc'],float(rate.group(1)),str(old_path.relative_to(ROOT))))
        if observed_rates:
            from openjev_dss.resource_planning import budgeted_runtime_minutes
            _,rate,rate_source=max(observed_rates)
            requested_minutes=args.minutes
            args.minutes=budgeted_runtime_minutes(float(balance_before.group(1)),rate,args.minutes)
            receipt.update(hard_stop_minutes=args.minutes,preallocation_budget={
                'requested_minutes':requested_minutes,'capped_minutes':args.minutes,'balance_cu':float(balance_before.group(1)),
                'observed_rate_cu_hr':rate,'rate_receipt':rate_source,'reserve_cu':2,
                'limitation':'Historical rate only; the fresh allocation quote must still pass the independent guard.'})
        if args.job=='localization_train':
            localization_policy=selected_localization_policy
            spent=0.
            for previous in (ROOT/'research/runs').glob('localization_train_*/receipt.json'):
                old=json.loads(previous.read_text())
                if not old.get('allocated_utc'):continue
                rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',old.get('usage_allocated',old.get('usage_at_allocation','')))
                assert rate,'Previous localization allocation is unpriced; inspect it before continuing'
                start=datetime.datetime.fromisoformat(old['allocated_utc'])
                end=datetime.datetime.fromisoformat(old.get('release_observed_utc',old.get('finished_utc',datetime.datetime.now(datetime.timezone.utc).isoformat())))
                spent+=max(0,(end-start).total_seconds())*float(rate.group(1))/3600
            probe_costs=ROOT/'research/localization_v1/optimization_probe_costs.json'
            if probe_costs.exists():spent+=json.loads(probe_costs.read_text())['sum_rate_time_estimate_cu']
            control_probe=ROOT/'research/localization_v2/control_connection_probe.json'
            if control_probe.exists():spent+=json.loads(control_probe.read_text()).get('rate_time_upper_estimate_cu',0.)
            training_limit=(continuation['legacy_training_cu_start']+continuation['additional_training_limit_cu']
                if continuation else localization_policy['resources']['training_cu_budget'])
            available=min(float(balance_before.group(1))-localization_policy['resources']['reserve_cu'],training_limit-spent)
            if args.localization_policy in ['research/localization_v3/policy.json','research/localization_v4/policy.json']:
                from openjev_dss.followup_budget import available as followup_available,spending
                available=min(available,followup_available(ROOT,'v3_training'))
                receipt['followup_budget']=spending(ROOT)
            if continuation:
                # Leave time for publication, settlement and observed release.
                available=max(0.,available-.2)
            assert observed_rates and available>0,'No localization credit allowance remains'
            args.minutes=min(args.minutes,int(available/max(observed_rates)[1]*60))
            assert args.minutes>=20,'Too little localization allocation time remains'
            receipt.update(hard_stop_minutes=args.minutes,localization_budget={
                'prior_rate_time_estimate_cu':spent,'available_cu':available,'total_budget_cu':training_limit})
        native_policy=configure_drive_environment(ROOT)
        if native_policy:
            health=native_policy['healthcheck']
            observed=json.loads(run(['rclone','lsjson',health['remote'],'--stat','--hash','--drive-root-folder-id',
                'YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR','--retries','1','--low-level-retries','1'],timeout=60).stdout)
            assert observed['ID']==health['id'] and observed['Size']==health['bytes'] and observed['Hashes']['md5']==health['md5']
            receipt['native_drive_healthcheck']={'passed_before_gpu_allocation':True,'quota_project':native_policy['quota_project'],
                'source_remote':'gdrive','policy_sha256':hashlib.sha256((ROOT/'research/native_drive_policy.json').read_bytes()).hexdigest()}
        config = configparser.ConfigParser()
        config.read(Path.home() / '.config/rclone/rclone.conf')
        scoped = configparser.ConfigParser()
        scoped['gdrive'] = dict(config['gdrive'])
        scoped['gdrive']['root_folder_id'] = 'YOUR_DRIVE_ROOT_FOLDER_ID'
        # Small-object single request and large-object chunk size have separate verified probes.
        scoped['gdrive']['upload_cutoff'] = '128Mi'
        scoped['gdrive']['chunk_size'] = '256Mi'
        secret_dir = Path(tempfile.mkdtemp(prefix='ucf-colab-secrets-'))
        secret_dir.chmod(0o700)
        secret = secret_dir / 'rclone.conf'
        with secret.open('w') as f:
            scoped.write(f)
        secret.chmod(0o600)
        payload = ROOT / '.colab-work' / (args.job + '_source.zip')
        if args.job=='localization_train':
            from openjev_dss.deployment_payload import build_localization_payload
            receipt['source_archive_bytes']=build_localization_payload(ROOT,payload,args.localization_policy,args.continuation)
        else:
            with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for folder in ['src', 'research']:
                    for path in (ROOT / folder).rglob('*'):
                        if path.is_file() and '__pycache__' not in path.parts and 'runs' not in path.parts and path.suffix in ['.py', '.json']:
                            z.write(path, path.relative_to(ROOT).as_posix())
        receipt['source_archive_sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
        source_archive = ROOT / 'research/source_archives' / (args.job + '_' + receipt['source_archive_sha256'][:12] + '.zip')
        source_archive.parent.mkdir(exist_ok=True)
        source_archive.write_bytes(payload.read_bytes())
        receipt['source_archive'] = str(source_archive.relative_to(ROOT))
        save()
        receipt['allocation_requested_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        cli('new', '-s', session, '--gpu', args.gpu, timeout=360)
        receipt['allocated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt['usage_at_allocation']=cli('usage').stdout
        early_rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_at_allocation'])
        early_balance=re.search(r'Current balance:\s*([0-9.]+)',receipt['usage_at_allocation'])
        if early_rate and early_balance and float(early_rate.group(1))>0:
            assert args.minutes*float(early_rate.group(1))/60<=float(early_balance.group(1))-2,'Requested allocation timeout exceeds the initial quoted credit budget'
        # Save the actual endpoint before any websocket operation can prune only
        # the CLI's local metadata; HTTP observation can then recover the same VM.
        control_observation=observe_worker('/content/ucf_mount_verified.json')
        if control_observation.get('endpoint'):
            receipt['endpoint']=control_observation['endpoint']
        save()
        if args.job=='localization_train' and args.localization_policy=='research/localization_v4/policy.json':
            interpreter=Path(shutil.which('colab')).resolve().with_name('python')
            receipt['kernel_initialization']=json.loads(run([str(interpreter),'scripts/initialize_colab_kernel.py',session,receipt['endpoint']],timeout=90).stdout)
            save()
        if args.job=='localization_train':
            timer=ROOT/'.colab-work/localization_watchdog.py'
            timer.write_text('import os\n'+f'os.environ["UCF_HARD_STOP_SECONDS"]={str(args.minutes*60)!r}\n'+
                (f'os.environ["RCLONE_HEADER"]={os.environ["RCLONE_HEADER"]!r}\nos.environ["UCF_DRIVE_QUOTA_PROJECT"]={native_policy["quota_project"]!r}\n' if native_policy else '')+
                (ROOT/'research/job_watchdog.py').read_text())
            execute(timer,30)
        else:
            cli('exec', '-s', session, '-f', 'research/job_watchdog.py', '--timeout', '30',
                '--env', f'UCF_HARD_STOP_SECONDS={args.minutes * 60}',
                *(['--env','RCLONE_HEADER='+os.environ['RCLONE_HEADER'],'--env','UCF_DRIVE_QUOTA_PROJECT='+native_policy['quota_project']] if native_policy else []),timeout=60)
        for local, remote in [(secret, '/content/ucf-rclone.conf'), (payload, '/content/ucf_project.zip'),
                               (ROOT / '.colab-work/fusermount3', '/content/ucf-fusermount3')]:
            if args.job=='localization_train':
                interpreter=Path(shutil.which('colab')).resolve().with_name('python')
                run([str(interpreter),'scripts/upload_colab_small.py',session,str(local),remote],timeout=180)
            else:cli('upload', '-s', session, str(local), remote)
        execute('research/bootstrap_transfer.py', 120)
        helper = ROOT / '.colab-work' / (args.job + '_configure.py')
        helper.write_text("import os\nfrom pathlib import Path\nif Path('/content/ucf-fusermount3').exists():Path('/content/ucf-fusermount3').replace('/content/ucf-bin/fusermount3')\nPath('/content/ucf-bin/fusermount3').chmod(0o755)\nos.environ['PATH']='/content/ucf-bin:'+os.environ['PATH']\n" +
                          f"os.environ['UCF_GROUP']={args.job!r}\nos.environ['UCF_SCALE_PROBE']={str(int(args.probe))!r}\nos.environ['UCF_DEADLINE']=str(UCF_JOB_HARD_DEADLINE-60)\nos.environ['UCF_FORMAL_EPOCH_LIMIT']={str(args.epoch_limit)!r}\nos.environ['UCF_LOCALIZATION_PHASE']={args.localization_phase!r}\nos.environ['UCF_LOCALIZATION_POLICY']={args.localization_policy!r}\n")
        if continuation:
            with helper.open('a') as stream:stream.write(f"os.environ['UCF_LOCALIZATION_CONTINUATION']={args.continuation!r}\n")
        execute(helper, 30)
        from openjev_dss.setup_recovery import execute_setup_with_evidence
        receipt['mount_execution']=execute_setup_with_evidence(
            lambda: execute('research/mount_rclone.py',90),
            lambda: observe_worker('/content/ucf_mount_verified.json'))
        save()
        mount_receipt = directory / 'mount.json'
        cli('download', '-s', session, '/content/ucf_mount_verified.json', str(mount_receipt))
        assert json.loads(mount_receipt.read_text())['test_videos'] == 290
        execute('research/setup_runtime.py', 300)
        environment = directory / 'environment.json'
        cli('download', '-s', session, '/content/ucf_environment_ready.json', str(environment))
        assert json.loads(environment.read_text())['ready'], 'Runtime versions differ from numerical preflight'
        if args.job == 'qwen_27b':
            assert json.loads(environment.read_text())['gpu_gib'] >= 64, '27B BF16 needs a larger GPU; do not download weights'
        receipt.update(status='running', usage_allocated=cli('usage').stdout)
        quoted_rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_allocated'])
        quoted_balance=re.search(r'Current balance:\s*([0-9.]+)',receipt['usage_allocated'])
        assert quoted_rate and quoted_balance and float(quoted_rate.group(1))>0,'No verified GPU price before model work'
        elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(receipt['allocated_utc'])).total_seconds()
        bound=max(0,args.minutes*60-elapsed)*float(quoted_rate.group(1))/3600
        receipt['credit_guard']={'balance_cu':float(quoted_balance.group(1)),'rate_cu_hr':float(quoted_rate.group(1)),
                                 'remaining_timeout_bound_cu':bound,'reserve_cu':2}
        assert bound<=float(quoted_balance.group(1))-2,'Requested remaining GPU timeout exceeds current credit budget; reduce the allocation duration before running models'
        if args.job=='localization_train':
            assert bound<=receipt['localization_budget']['available_cu'],'Fresh quote exceeds the remaining localization credit allowance'
            rate_helper=ROOT/'.colab-work/localization_rate.py'
            rate_helper.write_text("import os\nos.environ['UCF_QUOTED_RATE_CU_HR']="+repr(quoted_rate.group(1))+"\n")
            execute(rate_helper,30)
        initial_observation = observe_worker('/content/ucf_group_status.json')
        if initial_observation.get('endpoint'):
            receipt['endpoint'] = initial_observation['endpoint']
        save()
    print('RUNNING', args.job, session, str(directory), flush=True)
    remote_status = ('/content/ucf_scale_probe.json' if args.probe else '/content/ucf_localization_train/status.json' if args.job == 'localization_train' else '/content/ucf_formal_train/status.json' if args.job == 'train_formal' else '/content/ucf_qwen_mil_train/status.json' if args.job == 'train' else '/content/ucf_group_status.json')
    # A websocket interruption must not kill a healthy, hours-long worker.
    # The remote PID/start-time handle is idempotent and observed through HTTP.
    try:
        if not adopting:
            execute('research/launch_background_worker.py', 45)
    except Exception:
        observation = observe_worker(remote_status)
        if observation.get('worker_handle_http_status') == 404:
            busy = any(k.get('execution_state') == 'busy' for k in observation.get('kernels', []))
            if busy:
                time.sleep(10)
                observation = observe_worker(remote_status)
            if observation.get('worker_handle_http_status') == 404:
                execute('research/launch_background_worker.py', 45)
        elif observation.get('worker_alive') is False:
            raise
    hard_limit = time.time() + args.minutes * 60 + 120
    if adopting:
        hard_limit = datetime.datetime.fromisoformat(receipt['allocated_utc']).timestamp() + args.minutes * 60 + 120
    while time.time() < hard_limit:
        try:
            observation = observe_worker(remote_status)
            receipt['last_observation'] = observation
            save()
            if observation.get('assignment_present') is False:
                raise RuntimeError('Backend assignment ended; preserve and resume durable checkpoints')
            if observation.get('worker_alive') is False:
                break
            # Keep the kernel active and reap an exited child. A heartbeat failure
            # is an observation failure, never permission to restart the worker.
            if args.job=='localization_train' and args.localization_policy=='research/localization_v4/policy.json':
                execute('research/heartbeat_worker.py',15)
            else:
                cli('exec', '-s', session, '-f', 'research/heartbeat_worker.py', '--timeout', '15', timeout=35, check=False)
        except RuntimeError as error:
            if 'Backend assignment ended' in str(error):
                raise
            receipt['last_observation_error'] = clean(str(error))
            save()
        except Exception as error:
            receipt['last_observation_error'] = type(error).__name__
            save()
        time.sleep(20)
    else:
        raise RuntimeError('Explicit runtime hard-stop budget reached; inspect saved partial results')
    result_path = directory / 'result.json'
    cli('download', '-s', session, remote_status, str(result_path))
    result = json.loads(result_path.read_text())
    if args.probe:
        if result['status'] != 'passed':
            raise RuntimeError('Scale probe failed: ' + str(result.get('error', result['status'])))
    elif args.job == 'localization_train':
        assert result['test_used'] is False
        receipt['training_status']=result['status']
        for name in ['policy.json','contract.json','gradient_preflight.json','throughput_profile.json','normal_reference.json','selection_audit.json','baseline_validation.json','runtime_receipt.json','optimization_qualification.json','teacher_import.json','first_optimizer_update.json']+(['continuation_restore_audit.json','continuation_first_optimizer_update.json'] if continuation else []):
            cli('download','-s',session,'/content/ucf_localization_train/'+name,str(directory/name),timeout=90,check=False)
        if result['status'] not in ['complete','budget_complete','prepared']:
            raise RuntimeError('Localization training checkpointed but unfinished: '+str(result['status']))
    elif args.job == 'train_formal':
        policy = json.loads((ROOT / 'research/training_protocol_v2.json').read_text())
        assert result['test_used'] is False
        if result['status'] != 'complete':
            receipt['training_status'] = result['status']
            raise RuntimeError('Formal training incomplete: ' + str(result.get('error', result['status'])))
        for arm in result['arms'].values():
            assert arm['stop_reason'] == 'validation_plateau' and arm['stopped']
            assert len(arm['validation']) >= policy['selection']['min_epochs']
            assert all(v['videos'] == policy['validation']['videos'] for v in arm['validation'])
            assert arm['frozen_parameters_unchanged']
            assert arm['best_epoch'] == arm['selected_artifact']['epoch']
        # Fetch only small training evidence; weights and feature caches remain on Drive.
        for name in ['policy.json', 'contract.json', 'preflight.json', 'runtime_receipt.json']:
            cli('download', '-s', session, '/content/ucf_formal_train/' + name, str(directory / name))
    elif args.job == 'train':
        if not (result['status'] == 'complete' and result['steps'] == 160 and result['validation_videos'] == 163):
            raise RuntimeError('Training incomplete: ' + str(result.get('error', result['status'])))
    else:
        receipt['completed'] = result['completed']
        registry_name = 'formal_inference_jobs.json' if args.job == 'formal_mil' else 'inference_jobs.json'
        expected = [j['name'] for j in json.loads((ROOT / 'research' / registry_name).read_text())['groups'][args.job]]
        if not (result['status'] == 'complete' and result['completed'] == expected):
            raise RuntimeError('Group incomplete: ' + str(result.get('error', result['status'])))
    if args.job in ['formal_mil','openjev_zs']:
        try:
            cache_receipt=directory/'feature_prefetch.json'
            cli('download','-s',session,'/content/ucf_feature_prefetch.json',str(cache_receipt),timeout=60)
            json.loads(cache_receipt.read_text())
            receipt['feature_prefetch_receipt']=str(cache_receipt.relative_to(ROOT))
        except Exception as error:
            receipt['feature_prefetch_receipt_error']=type(error).__name__
    receipt['status'] = 'complete'
except Exception as error:
    receipt.update(status='incomplete', error=clean(type(error).__name__ + ': ' + str(error)))
    # Setup/import failures may happen before a status JSON exists. Keep the
    # actual worker traceback before releasing its ephemeral runtime.
    try:
        failure_log = directory / 'worker_failure.log'
        cli('download', '-s', session, '/content/ucf_worker.log', str(failure_log), timeout=45, check=False)
        if failure_log.exists():
            failure_log.write_text(clean(failure_log.read_text(errors='replace')))
            receipt['worker_failure_log'] = str(failure_log.relative_to(ROOT))
        transfer_log=directory/'transfer_failure.log'
        cli('download','-s',session,'/content/ucf_transfer_errors.log',str(transfer_log),timeout=45,check=False)
        if transfer_log.exists():
            transfer_log.write_text(clean(transfer_log.read_text(errors='replace')))
            receipt['native_transfer_failure_log']=str(transfer_log.relative_to(ROOT))
    except Exception:
        pass

    # A terminal exception never proves that the full group completed.
    try:
        result_path = directory / 'result.json'
        remote_status = ('/content/ucf_scale_probe.json' if args.probe else '/content/ucf_localization_train/status.json' if args.job == 'localization_train' else '/content/ucf_formal_train/status.json' if args.job == 'train_formal' else '/content/ucf_qwen_mil_train/status.json' if args.job == 'train' else '/content/ucf_group_status.json')
        cli('download', '-s', session, remote_status, str(result_path), timeout=45, check=False)
        if result_path.exists():
            receipt['completed'] = json.loads(result_path.read_text()).get('completed', [])
    except Exception:
        pass
finally:
    if receipt.get('allocated_utc') and not args.probe and args.job not in ['train','train_formal','localization_train']:
        for name in ['optional_cache_sync','optional_cache_policy']:
            try:
                target=directory/(name+'.json')
                cli('download','-s',session,'/content/ucf_'+name+'.json',str(target),timeout=45)
                json.loads(target.read_text())
                receipt[name+'_receipt']=str(target.relative_to(ROOT))
            except Exception as error:
                receipt[name+'_receipt_error']=type(error).__name__
    try:
        stopped = cli('stop', '-s', session, timeout=90, check=False)
        receipt.update(stop_exit_code=stopped.returncode, stop_output=stopped.stdout,
                       release_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       usage_after=cli('usage').stdout)
    except Exception as error:
        receipt['cleanup_error'] = clean(str(error))
    receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps(receipt, indent=2), flush=True)

# Pull only small, complete score bundles after releasing the GPU.
for name in receipt['completed']:
    raw = ROOT / 'results/controlled_raw' / (name + '.zip')
    raw.parent.mkdir(parents=True, exist_ok=True)
    run(['rclone', 'copyto', 'gdrive:remaining_20261108_v1/jobs/' + name + '/results.zip', str(raw),
         '--drive-root-folder-id', 'YOUR_DRIVE_ROOT_FOLDER_ID', '--log-level', 'ERROR'], timeout=180)
    destination = ROOT / 'results/controlled' / name
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw) as archive:
        assert archive.testzip() is None
        for entry in archive.namelist():
            assert (destination / entry).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    marker = json.loads((destination / 'complete.json').read_text())
    assert marker['complete'] and marker['videos'] == 290 and marker['windows'] == 37188
sys.exit(0 if receipt['status'] == 'complete' else 1)
