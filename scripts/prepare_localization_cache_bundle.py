"""Package existing cache on a bounded CPU runtime, release, then wake once."""
import configparser,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import configure_drive_environment,sanitize
from openjev_dss.followup_budget import available,spending
from openjev_dss.learning import atomic_json
from openjev_dss.desktop_events import call_desktop
folder_root=ROOT/'research/localization_v4_plateau'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=folder_root/'cache_cpu'/stamp;folder.mkdir(parents=True)
session='ucf-v4-cache-bundle';interpreter=Path(shutil.which('colab')).resolve().with_name('python')
receipt=dict(status='preparing',session=session,gpu='CPU',controller_pid=os.getpid(),started_utc=stamp,
    purpose='Byte-preserving cache packaging;no model inference or parameter updates',max_cu=.25,
    localization_policy='research/localization_v4/policy.json',continuation_control='research/localization_v4_plateau/control.json')
def save():atomic_json(folder/'receipt.json',receipt)
def run(args,timeout=90,check=True):
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=timeout)
    with (folder/'controller.log').open('a') as log:log.write(sanitize(p.stdout+p.stderr)+'\n')
    if check and p.returncode:raise RuntimeError('Cache CPU command failed:'+str(args[:3]))
    return p
def cli(*args,**kw):return run(['colab',*args],**kw)
def execute(source,timeout):
    p=folder/('cell_'+hashlib.sha256(source.encode()).hexdigest()[:12]+'.py');p.write_text(source)
    return run([interpreter,'scripts/execute_colab_control.py',session,receipt['endpoint'],p,'--timeout',str(timeout)],timeout=timeout+60)
secret_dir=None
try:
    save();configure_drive_environment(ROOT)
    assert available(ROOT,'v3_training')>=.25
    receipt['usage_before']=cli('usage').stdout
    assert 'Active assignments: 0' in receipt['usage_before']
    inventory_path=folder_root/'cache_inventory.json';inventory=json.loads(inventory_path.read_text())
    assert inventory['count']==len(inventory['files']) and inventory['bytes']==sum(r['bytes'] for r in inventory['files'])
    remote='gdrive:remaining_20261108_v1/localization_v4_plateau/cache_bundle_v1'
    # Refresh native OAuth before serializing the task-scoped config.
    run(['rclone','lsf',inventory['source_remote'],'--max-depth','1','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'])
    parser=configparser.ConfigParser();parser.read(Path.home()/'.config/rclone/rclone.conf')
    scoped=configparser.ConfigParser();scoped['gdrive']=dict(parser['gdrive']);scoped['gdrive']['root_folder_id']='YOUR_DRIVE_ROOT_FOLDER_ID'
    secret_dir=Path(tempfile.mkdtemp(prefix='cache-bundle-secrets-'));secret_dir.chmod(0o700);secret=secret_dir/'rclone.conf'
    with secret.open('w') as f:scoped.write(f)
    secret.chmod(0o600)
    archive=folder/'source.zip'
    paths=['research/bootstrap_assets.json','research/build_visual_cache_bundle.py','src/openjev_dss/cache_bundle.py',str(inventory_path.relative_to(ROOT))]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as pack:
        for path in paths:pack.write(ROOT/path,path)
    receipt['source_archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    cli('new','-s',session,timeout=180);receipt['allocated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt['usage_allocated']=cli('usage').stdout
    rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_allocated'])[1]);assert 0<=rate<=.1
    assert rate*2.5<=receipt['max_cu']
    info=json.loads(run([interpreter,'-c',"import json;from colab_cli.common import state;s=state.store.get("+repr(session)+");print(json.dumps(dict(endpoint=s.endpoint,kernel_id=s.kernel_id)))"],30).stdout)
    receipt['endpoint']=info['endpoint'];save()
    receipt['kernel_initialization']=json.loads(run([interpreter,'scripts/initialize_colab_kernel.py',session,receipt['endpoint']],90).stdout)
    watchdog="import os\nos.environ['UCF_HARD_STOP_SECONDS']='9000'\nos.environ['RCLONE_HEADER']="+repr(os.environ['RCLONE_HEADER'])+"\nos.environ['UCF_DRIVE_QUOTA_PROJECT']="+repr(os.environ['UCF_DRIVE_QUOTA_PROJECT'])+'\n'+(ROOT/'research/job_watchdog.py').read_text()
    execute(watchdog,30)
    config=dict(inventory=str(inventory_path.relative_to(ROOT)),inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),output_remote=remote,deadline=time.time()+8700)
    atomic_json(folder/'config.json',config)
    for source,target in [(secret,'/content/ucf-rclone.conf'),(archive,'/content/ucf_project.zip'),(folder/'config.json','/content/cache_bundle_config.json')]:
        run([interpreter,'scripts/upload_colab_small.py',session,source,target],120)
    execute((ROOT/'research/bootstrap_transfer.py').read_text(),120)
    execute("import zipfile,subprocess,sys,json,time\nfrom pathlib import Path\nPath('/content/ucf_project').mkdir(exist_ok=True)\nwith zipfile.ZipFile('/content/ucf_project.zip') as z:z.extractall('/content/ucf_project')\nh=Path('/content/cache_bundle_worker.json')\nif not h.exists():\n log=Path('/content/cache_bundle_worker.log').open('w')\n p=subprocess.Popen([sys.executable,'/content/ucf_project/research/build_visual_cache_bundle.py'],stdout=log,stderr=subprocess.STDOUT)\n h.write_text(json.dumps(dict(pid=p.pid,started=time.time())))\nprint(h.read_text())\n",30)
    receipt.update(status='running',output_remote=remote);save()
    while time.time()<config['deadline']-60:
        try:p=run([interpreter,'scripts/read_colab_remote.py',session,'/content/cache_bundle_status.json','--endpoint',receipt['endpoint'],'--tail','12000'],55,False)
        except subprocess.TimeoutExpired:
            receipt['last_observation_error']='Read timeout;not evidence of worker failure';save();time.sleep(30);continue
        for line in p.stdout.splitlines():
            if not line.startswith('{'):continue
            row=json.loads(line)
            if row.get('http_status')==200:
                receipt['progress']=json.loads(row['text']);save()
        progress=receipt.get('progress',{})
        if progress.get('status')=='complete':receipt['status']='complete';break
        if progress.get('status')=='error':raise RuntimeError('Remote packaging failed:'+progress.get('error',''))
        time.sleep(30)
    if receipt['status']!='complete':raise TimeoutError('CPU cache preparation time budget reached;verified parts retained remotely')
except Exception as exc:receipt.update(status='error',error=sanitize(str(exc))[-3000:])
finally:
    if receipt.get('allocated_utc'):
        try:receipt['stop_exit_code']=cli('stop','-s',session,timeout=90,check=False).returncode
        except Exception as exc:receipt['cleanup_error']=str(exc)
    try:
        receipt['usage_after']=cli('usage').stdout
        if 'Active assignments: 0' in receipt['usage_after']:receipt['release_observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    except Exception as exc:receipt['cleanup_error']=str(exc)
    if secret_dir:shutil.rmtree(secret_dir)
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if receipt['status']=='complete':
    run(['rclone','copyto',remote+'/manifest.json',folder_root/'cache_bundle_manifest.json','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],120)
    assert hashlib.sha256((folder_root/'cache_bundle_manifest.json').read_bytes()).hexdigest()==receipt['progress']['manifest_sha256']
event_path=folder_root/'monitor/events'/('cache_bundle_'+stamp+'.json')
event=dict(id=event_path.stem,kind='cache_bundle_'+receipt['status'],receipt=str((folder/'receipt.json').relative_to(ROOT)),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary='CPU cache packaging settled;check verified manifest and resource release,then resume same806-step trajectory on one L4 within existing20CU;no new training route')
atomic_json(event_path,event)
thread=json.loads((folder_root/'launch.json').read_text())['thread']
while not json.loads(event_path.read_text()).get('handled_utc'):
    try:
        reply=call_desktop('read_thread',dict(threadId=thread,turnLimit=1,includeOutputs=False,maxOutputCharsPerItem=0),thread,event['id']+'-idle')
        snap=json.loads(next(x['text'] for x in reply['contentItems'] if x['type']=='inputText'))
        if any(t.get('status')=='inProgress' for t in snap.get('turns',[])):time.sleep(15);continue
        call_desktop('send_message_to_thread',dict(threadId=thread,prompt=f"[缓存恢复事件 {event['id']}] {event['summary']}\n证据：{event_path}；收据：{folder/'receipt.json'}。先核验CPU已释放与是否存在任何活动分配。完成缓存SHA/逐文件MD5校验后，从806步完整AdamW/RNG断点恢复同一训练，禁止重复分配；仅L4、最多新增20CU含故障及CPU准备、阶段100CU、三小时每轮和原窗口独立损失/抽样/验证不变；原合格开发帧级AUROC平台期规则及最终唯一模型测试授权见research/localization_v4_plateau/control.json。必须核验807步或更后的真实参数变化；本次GPU启动前若最终包尚不完整先修复。更新本事件handled_utc/action_summary并结束回合，继续事件监控。"),thread,event['id'])
        event.update(delivery='delivered',delivered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic_json(event_path,event);break
    except (TimeoutError,EOFError):
        event.update(delivery='ambiguous_no_duplicate');atomic_json(event_path,event);break
    except Exception as exc:
        event.update(delivery='retry_pending',delivery_error=sanitize(str(exc))[-1000:]);atomic_json(event_path,event);time.sleep(30)
print(json.dumps(dict(status=receipt['status'],receipt=str((folder/'receipt.json').relative_to(ROOT)),event=str(event_path.relative_to(ROOT)))),flush=True)
