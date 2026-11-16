"""Cheap CPU-only test of refreshed Colab CLI execution and completion evidence."""
import subprocess,sys,json,datetime,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize
session='ucf-control-probe';record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_allocated=False,attempts=[])
folder=ROOT/'research/localization_v2';allocated=False

def cli(*args,timeout=60):
 p=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout)
 with (ROOT/'.colab-work/control_probe.log').open('a') as f:f.write(str(args[:3])+'\n'+sanitize(p.stdout+p.stderr)+'\n')
 return p
try:
 assert 'Active assignments: 0' in cli('usage').stdout
 p=cli('new','-s',session,timeout=150);assert p.returncode==0;allocated=True
 rate=cli('usage').stdout;record['usage_allocated']=rate
 import re
 assert float(re.search(r'Usage rate:\s*([0-9.]+)/hr',rate)[1])<=.1
 for index in range(3):
  script=ROOT/'.colab-work/control_probe_exec.py'
  script.write_text("from pathlib import Path\nimport json\np=Path('/content/ucf_control_probe.json')\nd=json.loads(p.read_text()) if p.exists() else {}\nd["+repr(str(index))+"]=True\np.write_text(json.dumps(d))\nprint('CONTROL_PROBE_EXECUTED',"+str(index)+")\n")
  for attempt in range(4):
   cli('sessions') # Official CLI sync_sessions refreshes proxy URL/token unconditionally.
   p=cli('exec','-s',session,'-f',str(script),'--timeout','20',timeout=50)
   observer=subprocess.run(['/opt/colab-cli/bin/python',str(ROOT/'scripts/read_colab_remote.py'),session,'/content/ucf_control_probe.json'],capture_output=True,text=True,timeout=35)
   good=False
   for line in observer.stdout.splitlines():
    try:r=json.loads(line);good=r.get('http_status')==200 and json.loads(r.get('text','{}')).get(str(index)) is True
    except ValueError:pass
    if good:break
   record['attempts'].append(dict(operation=index,attempt=attempt,cli_exit=p.returncode,remote_completion_verified=good))
   if good:break
   time.sleep(min(20,5*(attempt+1)))
  else:raise RuntimeError('CPU control-path probe did not complete')
 record['passed']=True
except Exception as exc:record.update(passed=False,error=sanitize(str(exc))[-1000:])
finally:
 if allocated:
  p=cli('stop','-s',session,timeout=90);record['stop_exit_code']=p.returncode
 record['usage_after']=cli('usage').stdout
 record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (folder/'control_connection_probe.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,ensure_ascii=False),flush=True)
