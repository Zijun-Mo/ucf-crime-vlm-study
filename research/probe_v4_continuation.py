"""Read-only diagnosis of the already running continuation worker."""
import json,os,subprocess,time
from pathlib import Path
out=Path('/content/ucf_localization_train')
report=dict(observed=time.time())
handle=json.loads(Path('/content/ucf_background_handle.json').read_text());pid=handle['pid']
report['worker']=handle
for field in ['status','wchan','io']:
    p=Path('/proc')/str(pid)/field
    report[field]=p.read_text() if p.exists() else 'missing'
rows=subprocess.run(['ps','-eo','pid,ppid,etimes,time,stat,comm,args'],capture_output=True,text=True,timeout=5).stdout.splitlines()
children={pid};selected=[]
for _ in range(3):
    for row in rows[1:]:
        bits=row.split(None,6)
        if len(bits)==7 and (int(bits[0]) in children or int(bits[1]) in children):
            children.add(int(bits[0]))
            if row not in selected:selected.append(row)
report['processes']=selected
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
report['files']={}
for name in ['status.json','throughput_profile.json','throughput_profile_before_continuation.json','optimization_qualification.json','native_teacher_cache_gate.json','continuation_restore_audit.json','continuation_first_optimizer_update.json','worker_error.txt']:
    p=out/name
    if p.exists():
        entry=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
        if name!='status.json':entry['text']=p.read_text()[-10000:]
        report['files'][name]=entry
cache=Path('/content/ucf_formal_visual');files=list(cache.rglob('*.pt'))
report['visual_cache']=dict(shards=len(files),bytes=sum(p.stat().st_size for p in files))
Path('/content/continuation_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print('CONTINUATION_DIAGNOSTIC_WRITTEN')
