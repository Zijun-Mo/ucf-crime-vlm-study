"""Prepare a byte-identical train/val canonical ZIP during the existing GPU job."""
import hashlib,json,subprocess,time,traceback,zipfile
from pathlib import Path
started=time.time();out=Path('/content/ucf_formal_input_pack_status.json');report={'status':'running','stage':'copy','new_gpu_allocation':False}
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--log-level','ERROR','--contimeout','15s','--timeout','60s','--retries','3']
def save():
    report['elapsed_s']=time.time()-started
    p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2)+'\n');p.replace(out)
try:
    payload=Path('/content/ucf_formal_metadata_bundle.json');bundle=json.loads(payload.read_text());records=bundle['records'];assert len(records)==1419
    data=Path('/content/ucf_formal_input_pack');data.mkdir(exist_ok=True);local=data/'canonical';local.mkdir(exist_ok=True)
    listing=data/'files.txt';listing.write_text('\n'.join(sorted(records))+'\n');save()
    subprocess.run(rc+['copy','gdrive:weak_dss_smoke_v1/canonical',str(local),'--files-from',str(listing),'--transfers','8','--checkers','8'],check=True,timeout=2400)
    archive=data/'canonical_train_val.zip';report['stage']='verify_and_pack';save()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for name in sorted(records):
            assert Path(name).name==name
            p=local/name
            with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==records[name]['canonical_sha256']
            z.write(p,name)
    with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    with archive.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
    remote='gdrive:remaining_20261108_v1/formal_v2/input/canonical_'+sha[:16]+'.zip'
    report.update(stage='upload',videos=len(records),zip_bytes=archive.stat().st_size,zip_sha256=sha,zip_md5=md5,
                  metadata_bundle_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),remote=remote);save()
    subprocess.run(rc+['copyto',str(archive),remote],check=True,timeout=1200)
    server=json.loads(subprocess.run(rc+['lsjson',remote,'--stat','--hash'],check=True,capture_output=True,text=True,timeout=90).stdout)
    assert server['Size']==archive.stat().st_size and server['Hashes']['md5']==md5
    report.update(status='passed',stage='complete',server_id=server.get('ID'),original_files_changed=False,desktop_video_download_bytes=0);save()
except Exception:
    report.update(status='error',error=traceback.format_exc());save();raise
