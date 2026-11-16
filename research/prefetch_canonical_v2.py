"""Bounded, read-only FUSE prefetch that follows the current inference condition."""
import concurrent.futures
import json
from pathlib import Path
import time

root=Path('/content/ucf_drive')
handle=json.loads(Path('/content/ucf_background_handle.json').read_text())
records=json.loads((root/'formal_stride1/manifest.json').read_text())['videos']
destination=Path('/content/ucf_prefetch_status_v2.json')

def alive():
    try:
        f=(Path('/proc')/str(handle['pid'])/'stat').read_text().split(') ',1)[1].split()
        return f[0]!='Z' and f[19]==handle['start_ticks']
    except FileNotFoundError:return False


def warm(record):
    size=0
    with (root/'formal_stride1/canonical'/record['name']).open('rb') as f:
        while block:=f.read(8*1024**2):size+=len(block)
    return size

seen=set();pending={};warmed=total=0;active=None;errors=0
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    while alive():
        group=json.loads(Path('/content/ucf_group_status.json').read_text())
        if group.get('status') in ['complete','error']:break
        for key,future in list(pending.items()):
            if future.done():
                try:total+=future.result();warmed+=1
                except Exception:seen.discard(key);errors+=1
                del pending[key]
        active=group.get('active_job')
        if active:
            completed={p.name.removesuffix('.json') for p in (Path('/content/ucf_results')/active/'predictions').glob('*.json')}
            remaining=[r for r in records if r['name'] not in completed]
            for record in remaining[:4]:
                key=(active,record['name'])
                if key not in seen and len(pending)<2:
                    pending[key]=pool.submit(warm,record);seen.add(key)
        report={'active_job':active,'warmed':warmed,'bytes':total,'pending':[list(k) for k in pending],
                'errors':errors,'worker_pid':handle['pid'],'worker_start_ticks':handle['start_ticks'],
                'updated':time.time(),'read_only':True,'max_workers':2}
        temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(report));temporary.replace(destination)
        time.sleep(2)
