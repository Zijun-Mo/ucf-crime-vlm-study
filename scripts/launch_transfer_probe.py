"""Validate large uploads on a bounded CPU Colab VM, then always release it."""
import configparser,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
session='ucf-large-upload-cpu';directory=ROOT/'research/runs'/('transfer_cpu_'+stamp);directory.mkdir(parents=True)
receipt={'job':'large_upload_cpu_probe','session':session,'gpu':'NONE','started_utc':stamp,'status':'preparing','controller_pid':os.getpid()}
def save():(directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
def run(command,timeout=90,check=True):
    r=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    with (directory/'controller.log').open('a') as log:log.write('COMMAND '+str(command[:4])+'\n'+sanitize(r.stdout+r.stderr)+'\n')
    if check and r.returncode:raise RuntimeError('Command failed: '+str(command[:4]))
    return r
def cli(*command,**kwargs):return run(['colab',*command],**kwargs)
try:
    save();receipt['usage_before']=cli('usage').stdout
    assert re.search(r'Active assignments:\s*0',receipt['usage_before']),'Release the previous assignment before CPU probe'
    cfg=configparser.ConfigParser();cfg.read(Path.home()/'.config/rclone/rclone.conf')
    scoped=configparser.ConfigParser();scoped['gdrive']=dict(cfg['gdrive']);scoped['gdrive']['root_folder_id']='YOUR_DRIVE_ROOT_FOLDER_ID'
    private=Path(tempfile.mkdtemp(prefix='ucf-transfer-secrets-'));private.chmod(0o700);secret=private/'rclone.conf'
    with secret.open('w') as stream:scoped.write(stream)
    secret.chmod(0o600)
    cli('new','-s',session);receipt['allocated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    cli('exec','-s',session,'-f','research/job_watchdog.py','--env','UCF_HARD_STOP_SECONDS=900','--timeout','30',timeout=60)
    bootstrap=ROOT/'.colab-work/transfer_bootstrap.zip'
    with zipfile.ZipFile(bootstrap,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT/'research/bootstrap_assets.json','research/bootstrap_assets.json')
    for local,remote in [(bootstrap,'/content/ucf_project.zip'),(secret,'/content/ucf-rclone.conf'),(ROOT/'research/large_transfer_probe.py','/content/large_transfer_probe.py'),(ROOT/'src/openjev_dss/cloud_io.py','/content/cloud_io.py')]:
        cli('upload','-s',session,str(local),remote)
    cli('exec','-s',session,'-f','research/bootstrap_transfer.py','--timeout','120',timeout=150)
    cli('download','-s',session,'/content/ucf_transfer_preflight.json',str(directory/'bootstrap.json'))
    assert json.loads((directory/'bootstrap.json').read_text())['binary_sha256']=='ed7ab1f17876bb006c3df3949a3044959671afd9112362dacaa5dd1647735bc1'
    receipt['usage_allocated']=cli('usage').stdout;save()
    helper=ROOT/'.colab-work/start_large_transfer.py'
    helper.write_text("import os,json,subprocess,sys,time\nfrom pathlib import Path\nos.environ['UCF_DEADLINE']=str(UCF_JOB_HARD_DEADLINE-30)\nlog=open('/content/ucf_worker.log','w')\nUCF_BACKGROUND_JOB=subprocess.Popen([sys.executable,'/content/large_transfer_probe.py'],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())\nf=Path('/proc')/str(UCF_BACKGROUND_JOB.pid)/'stat'\nfields=f.read_text().split(') ',1)[1].split()\nPath('/content/ucf_background_handle.json').write_text(json.dumps({'pid':UCF_BACKGROUND_JOB.pid,'start_ticks':fields[19],'script':'large_transfer_probe.py','group':'cpu_transfer_probe','started':time.time()}))\n")
    cli('exec','-s',session,'-f',str(helper),'--timeout','30',timeout=60)
    receipt['status']='running';save();print('CPU_UPLOAD_PROBE_RUNNING',str(directory),flush=True)
    interpreter=str(Path(shutil.which('colab')).resolve().with_name('python'))
    for _ in range(30):
        result=run([interpreter,'scripts/inspect_colab_job.py',session,'--worker','--ssh','--progress','/content/ucf_large_transfer_probe.json'],timeout=90)
        observation=json.loads(result.stdout);receipt['last_observation']=observation
        if observation.get('endpoint'):receipt['endpoint']=observation['endpoint']
        save()
        if observation.get('worker_alive') is False:break
        if observation.get('assignment_present') is False:raise RuntimeError('CPU backend ended; preserve any probe report')
        cli('exec','-s',session,'-f','research/heartbeat_worker.py','--timeout','15',timeout=35,check=False)
        time.sleep(15)
    cli('download','-s',session,'/content/ucf_large_transfer_probe.json',str(directory/'result.json'))
    result=json.loads((directory/'result.json').read_text());assert result['status']=='passed',str(result)
    (ROOT/'research/large_transfer_probe_result.json').write_text(json.dumps(result,indent=2)+'\n')
    receipt['status']='complete'
except Exception as error:receipt.update(status='incomplete',error=sanitize(str(error)))
finally:
    try:
        stopped=cli('stop','-s',session,timeout=90,check=False);receipt.update(stop_exit_code=stopped.returncode,usage_after=cli('usage').stdout)
    except Exception as error:receipt['cleanup_error']=sanitize(str(error))
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(receipt,indent=2),flush=True)
