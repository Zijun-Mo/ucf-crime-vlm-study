"""One bounded CPU-only case-evidence job; never download video/model files."""
import argparse,configparser,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import configure_drive_environment,sanitize
from openjev_dss.followup_budget import available,spending
from openjev_dss.learning import atomic_json
parser_args=argparse.ArgumentParser();parser_args.add_argument('--epoch',type=int,choices=[2,5,9],default=2);args=parser_args.parse_args()
experiment={2:'localization_v4_test',5:'localization_v4_epoch5_test',9:'localization_v4_plateau_test'}[args.epoch]
result_name='qwen_localization_v4_plateau_test' if args.epoch==9 else f'qwen_localization_v4_epoch{args.epoch}_test'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=ROOT/f'research/{experiment}/analysis_cpu'/stamp;folder.mkdir(parents=True)
session=f'ucf-v4-e{args.epoch}-case-evidence';interpreter=Path(shutil.which('colab')).resolve().with_name('python')
r=dict(status='preparing',controller_pid=os.getpid(),session=session,gpu='CPU',started_utc=stamp,purpose='fixed-case image evidence; no inference or annotation changes',max_cu=.05)
def save():atomic_json(folder/'receipt.json',r)
def run(cmd,timeout=60,check=True):
    p=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,timeout=timeout)
    with (folder/'controller.log').open('a') as log:log.write(sanitize(p.stdout+p.stderr)+'\n')
    if check and p.returncode:raise RuntimeError('Case CPU control failed; inspect sanitized log')
    return p
def cli(*args,**kw):return run(['colab',*args],**kw)
def execute(source,timeout):
    file=folder/('cell_'+hashlib.sha256(source.encode()).hexdigest()[:12]+'.py');file.write_text(source)
    return run([interpreter,'scripts/execute_colab_control.py',session,r['endpoint'],file,'--timeout',str(timeout)],timeout=timeout+60)
secret_dir=None
try:
    save();configure_drive_environment(ROOT);assert available(ROOT,'annotation_cpu')>=.05
    assert 'Active assignments: 0' in cli('usage').stdout
    policy=json.loads((ROOT/f'research/{experiment}/policy.json').read_text());cases=json.loads((ROOT/f'results/{result_name}/case_selection.json').read_text())['cases']
    records={row['name']:row for row in policy['records']};items=[]
    for role in ['worst','median','best']:
        name=cases[role]['name'];centers=[]
        for source in [ROOT/policy['baseline_source'],ROOT/f'results/{result_name}/raw/predictions']:
            p=json.loads((source/(name+'.json')).read_text());centers.append(float(p['start_s'][int(np.argmax(p['raw_logits']))]+2))
        items.append(dict(role=role,record=records[name],metadata=policy['metadata'][name],peak_window_centers_s=centers))
    config=dict(cases=items,canonical_remote=policy['canonical_remote'],output_remote=policy['output_remote']+'/case_frames')
    atomic_json(folder/'config.json',config)
    # Native rclone refreshes its cached OAuth token before we serialize the
    # scoped configuration used by the remote bootstrap's direct Drive read.
    run(['rclone','lsf','gdrive:','--max-depth','1','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],timeout=60)
    parser=configparser.ConfigParser();parser.read(Path.home()/'.config/rclone/rclone.conf');scoped=configparser.ConfigParser();scoped['gdrive']=dict(parser['gdrive']);scoped['gdrive']['root_folder_id']='YOUR_DRIVE_ROOT_FOLDER_ID'
    secret_dir=Path(tempfile.mkdtemp(prefix='v4-case-secrets-'));secret_dir.chmod(0o700);secret=secret_dir/'rclone.conf'
    with secret.open('w') as f:scoped.write(f)
    secret.chmod(0o600)
    archive=folder/'source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as pack:pack.write(ROOT/'research/bootstrap_assets.json','research/bootstrap_assets.json')
    r['source_sha256']=hashlib.sha256((ROOT/'research/render_v4_case_frames.py').read_bytes()).hexdigest();save()
    cli('new','-s',session,timeout=120);r['allocated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['usage_allocated']=cli('usage').stdout
    rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',r['usage_allocated'])[1]);assert 0<rate<=.1
    info=json.loads(run([interpreter,'-c',"import json;from colab_cli.common import state;s=state.store.get("+repr(session)+");print(json.dumps(dict(endpoint=s.endpoint,kernel_id=s.kernel_id)))"],timeout=30).stdout);r['endpoint']=info['endpoint'];save()
    r['kernel_initialization']=json.loads(run([interpreter,'scripts/initialize_colab_kernel.py',session,r['endpoint']],timeout=90).stdout)
    watchdog="import os\nos.environ['UCF_HARD_STOP_SECONDS']='1200'\nos.environ['RCLONE_HEADER']="+repr(os.environ['RCLONE_HEADER'])+"\nos.environ['UCF_DRIVE_QUOTA_PROJECT']="+repr(os.environ['UCF_DRIVE_QUOTA_PROJECT'])+'\n'+(ROOT/'research/job_watchdog.py').read_text()
    execute(watchdog,30)
    for source,target in [(secret,'/content/ucf-rclone.conf'),(archive,'/content/ucf_project.zip'),(folder/'config.json','/content/v4_case_config.json')]:run([interpreter,'scripts/upload_colab_small.py',session,source,target],timeout=100)
    execute((ROOT/'research/bootstrap_transfer.py').read_text(),120)
    execute((ROOT/'research/render_v4_case_frames.py').read_text(),240)
    r['status']='complete'
except Exception as exc:r.update(status='error',error=sanitize(str(exc))[-1500:])
finally:
    if r.get('allocated_utc'):
        try:r['stop_exit_code']=cli('stop','-s',session,timeout=90,check=False).returncode
        except Exception as exc:r['cleanup_error']=str(exc)
    try:
        r['usage_after']=cli('usage').stdout
        if 'Active assignments: 0' in r['usage_after']:r['release_observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    except Exception as exc:r['cleanup_error']=str(exc)
    if secret_dir:shutil.rmtree(secret_dir)
    r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if r['status']=='complete':
    destination=ROOT/f'results/{result_name}/case_frames';destination.mkdir(exist_ok=True)
    run(['rclone','copy',config['output_remote'],destination,'--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],timeout=90)
    manifest=json.loads((destination/'manifest.json').read_text());assert len(manifest['cases'])==3
    for case in manifest['cases']:assert hashlib.sha256((destination/case['image']).read_bytes()).hexdigest()==case['image_sha256']
    r['local_image_hashes_verified']=True;save()
print(json.dumps(r,ensure_ascii=False),flush=True)
