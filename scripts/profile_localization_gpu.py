"""Bounded L4 precision screening; cannot start full localization training."""
import configparser,datetime,hashlib,json,os,re,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize,configure_drive_environment
session='ucf-localization-precision-probe';folder=ROOT/'research/localization_v1'/('precision_probe_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir(exist_ok=True)
receipt={'status':'preparing','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'session':session,'gpu':'L4','hard_stop_minutes':30,'full_training_started':False,'test_used':False}
secret=None

def save():
 p=folder/'receipt.tmp';p.write_text(json.dumps(receipt,indent=2)+'\n');p.replace(folder/'receipt.json')
def call(args,timeout=60,check=True):
 p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
 with (folder/'controller.log').open('a') as f:f.write('\nCOMMAND '+str(args[:4])+'\n'+sanitize(p.stdout+p.stderr)+'\n')
 if check and p.returncode:raise RuntimeError('Command failed '+str(args[:4])+' exit='+str(p.returncode))
 return p

def cli(*args,**kw):return call(['colab',*map(str,args)],**kw)
def execute(path,timeout=60):
 for attempt in range(3):
  try:return cli('exec','-s',session,'-f',path,'--timeout',timeout,timeout=timeout+40)
  except (RuntimeError,subprocess.TimeoutExpired):
   if attempt==2:raise
   time.sleep(2)

try:
 configure_drive_environment(ROOT)
 call(['rclone','lsf','gdrive:','--max-depth','1','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--log-level','ERROR'],timeout=80)
 usage=cli('usage').stdout;receipt['usage_before']=usage;save()
 assert re.search(r'Active assignments:\s*0',usage),'Release previous assignment before bounded hardware probe'
 assert float(re.search(r'Current balance:\s*([0-9.]+)',usage)[1])>20
 config=configparser.ConfigParser();config.read(Path.home()/'.config/rclone/rclone.conf')
 scoped=configparser.ConfigParser();scoped['gdrive']=dict(config['gdrive']);scoped['gdrive']['root_folder_id']='YOUR_DRIVE_ROOT_FOLDER_ID'
 temp=Path(tempfile.mkdtemp(prefix='localization-probe-'));temp.chmod(0o700);secret=temp/'rclone.conf'
 with secret.open('w') as f:scoped.write(f)
 secret.chmod(0o600)
 archive=ROOT/'.colab-work/localization_probe_source.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in (ROOT/'src').rglob('*.py'):z.write(p,p.relative_to(ROOT))
  for name in ['protocol.json','gpu_preflight_result.json','bootstrap_assets.json','formal_training_metadata.json','benchmark_localization_lowprecision.py']:
   z.write(ROOT/'research'/name,'research/'+name)
 receipt['source_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();receipt['allocation_requested_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 cli('new','-s',session,'--gpu','L4',timeout=180)
 receipt['allocated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt['usage_allocated']=cli('usage').stdout;save()
 rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_allocated'])[1]);assert rate*.5<=1.2,'Bounded L4 precision probe quote exceeds 1.2CU'
 watchdog=ROOT/'.colab-work/precision_watchdog.py'
 watchdog.write_text('import os\nos.environ["UCF_HARD_STOP_SECONDS"]="1800"\nos.environ["RCLONE_HEADER"]='+repr(os.environ['RCLONE_HEADER'])+'\nos.environ["UCF_DRIVE_QUOTA_PROJECT"]='+repr(os.environ['UCF_DRIVE_QUOTA_PROJECT'])+'\n'+(ROOT/'research/job_watchdog.py').read_text())
 execute(watchdog,30)
 for src,dst in [(secret,'/content/ucf-rclone.conf'),(archive,'/content/ucf_project.zip')]:
  interp=Path(subprocess.check_output(['which','colab'],text=True).strip()).resolve().with_name('python')
  call([str(interp),'scripts/upload_colab_small.py',session,str(src),dst],timeout=180)
 execute('research/bootstrap_transfer.py',120);execute('research/setup_runtime.py',300)
 helper=ROOT/'.colab-work/localization_precision_start.py'
 helper.write_text('''import subprocess,sys,os,json,time
from pathlib import Path
code="""from pathlib import Path
import sys,subprocess,os
sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','2']
Path('/content/ucf_drive/weak_dss_learning_10cu_v1').mkdir(parents=True,exist_ok=True)
for remote,local in [('weak_dss_learning_10cu_v1/manifest.json','/content/ucf_drive/weak_dss_learning_10cu_v1/manifest.json'),('remaining_20261108_v1/localization_v1/optimization/localization_probe_fixtures.pt','/content/localization_probe_fixtures.pt')]:
 subprocess.run(rc+['copyto','gdrive:'+remote,local],check=True,timeout=300)
exec(compile(Path('/content/ucf_project/research/benchmark_localization_lowprecision.py').read_text(),'benchmark_localization_lowprecision.py','exec'))
"""
Path('/content/localization_precision_probe.py').write_text(code)
log=open('/content/ucf_localization_precision_probe.log','w')
handle=Path('/content/localization_precision_probe_handle.json')
if handle.exists():
 old=json.loads(handle.read_text());stat=Path('/proc')/str(old['pid'])/'stat'
 if stat.exists() and stat.read_text().split(') ',1)[1].split()[0]!='Z':
  print('PRECISION_PROBE_ALREADY_STARTED',old['pid'])
 else:raise RuntimeError('Prior precision worker exited; inspect its evidence before retry')
else:
 p=subprocess.Popen([sys.executable,'/content/localization_precision_probe.py'],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
 handle.write_text(json.dumps({'pid':p.pid,'started':time.time()}))
 print('PRECISION_PROBE_STARTED',p.pid)
''')
 execute(helper,45);receipt['status']='running';save();print('PRECISION_PROBE_RUNNING',flush=True)
 limit=time.monotonic()+1000
 while time.monotonic()<limit:
  interp=Path(subprocess.check_output(['which','colab'],text=True).strip()).resolve().with_name('python')
  obs=call([str(interp),'scripts/read_colab_remote.py',session,'/content/ucf_localization_optimization_precision/status.json','/content/ucf_localization_precision_probe.log','--tail','40000'],timeout=65,check=False)
  for line in obs.stdout.splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('http_status')!=200:continue
   if r['path'].endswith('status.json'):
    value=json.loads(r['text']);(folder/'benchmark.json').write_text(json.dumps(value,indent=2)+'\n')
    if value['status'] in ['benchmarked','error']:
     receipt['status']=value['status'];save();print('PRECISION_PROBE_RESULT',json.dumps({'status':value['status'],'best_training_hours':(value.get('best') or {}).get('training_hours_estimate')}),flush=True);raise SystemExit(0 if value['status']=='benchmarked' else 1)
   else:(folder/'worker.log').write_text(sanitize(r['text']))
  time.sleep(20)
 raise TimeoutError('Hardware screening did not finish within its bounded window')
except Exception as e:
 receipt.update(status='failed',error=sanitize(str(e)));print('PRECISION_PROBE_FAILED',type(e).__name__,str(e),flush=True)
finally:
 if receipt.get('allocation_requested_utc'):
  try:
   p=cli('stop','-s',session,timeout=90,check=False);receipt['stop_exit_code']=p.returncode;receipt['release_observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt['usage_after']=cli('usage').stdout
  except Exception as e:receipt['cleanup_error']=type(e).__name__
 if secret:
  secret.unlink(missing_ok=True);secret.parent.rmdir()
 receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
