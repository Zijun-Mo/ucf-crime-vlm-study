"""CPU-only publication probe; never a model feature or experiment result."""
import hashlib,json,os,sys,time,traceback,zipfile
from pathlib import Path
started=time.time();out=Path('/content/ucf_optional_cache_probe.json')
report={'status':'running','new_gpu_allocation':False,'model_forward_calls':0,'desktop_video_download_bytes':0}
try:
    package=Path('/content/ucf_io_probe_package');package.mkdir(exist_ok=True)
    with zipfile.ZipFile('/content/optional_cache_probe_source.zip') as z:z.extractall(package)
    sys.path.insert(0,str(package))
    from openjev_dss.optional_cache_sync import OptionalCacheSync
    data=Path('/content/ucf_io_probe');data.mkdir(exist_ok=True);source=data/'transport_probe.pt'
    with source.open('wb') as stream:stream.truncate(3601*49*2048*2)
    command=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--header',
             'X-Goog-User-Project: YOUR_GOOGLE_QUOTA_PROJECT','--contimeout','15s','--timeout','60s',
             '--retries','2','--low-level-retries','2','--stats','30s','--log-level','INFO']
    sync=OptionalCacheSync(command,'gdrive:remaining_20261108_v1/preflight/optional_cache_sync_cpu',
        '/content/ucf_optional_cache_probe_transfers.json',time.time()+300)
    before=time.monotonic();sync.submit(source);enqueue=time.monotonic()-before
    # Stand-in CPU work verifies the caller proceeds while native I/O is active.
    assert sum(range(100000))==4999950000
    sync.close(grace=180)
    actual=json.loads(Path('/content/ucf_optional_cache_probe_transfers.json').read_text())
    assert actual['status']=='complete' and actual['entries'][0]['status']=='verified',actual
    assert enqueue<.1
    report.update(status='passed',enqueue_s=enqueue,bytes=source.stat().st_size,
                  source_sha256=hashlib.sha256((package/'openjev_dss/optional_cache_sync.py').read_bytes()).hexdigest(),
                  transfer=actual['entries'][0],quota_project='YOUR_GOOGLE_QUOTA_PROJECT')
except Exception:report.update(status='error',error=traceback.format_exc())
report['elapsed_s']=time.time()-started;out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
