"""CPU-only chunked-upload check; large fixture stays exclusively in cloud runtimes."""
import hashlib,json,os,time,re
from pathlib import Path
from cloud_io import checked,sanitize,transfer_timeout
begin=time.time();out=Path('/content/ucf_large_transfer_probe.json')
report={'status':'running','started':begin,'cpu_only':True,'new_oauth_consent_needed':False,'trials':[]}
def save():out.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    fixture=Path('/content/ucf_large_transfer_fixture.bin')
    md5,sha=hashlib.md5(),hashlib.sha256()
    with fixture.open('wb') as stream:
        for _ in range(64):
            block=os.urandom(4*1024**2);stream.write(block);md5.update(block);sha.update(block)
    report.update(fixture_bytes=fixture.stat().st_size,fixture_sha256=sha.hexdigest(),fixture_md5=md5.hexdigest())
    rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','2']
    for chunk in [64,256]:
        remote=f'gdrive:remaining_20261108_v1/preflight/io_bench/large_{int(begin)}_{chunk}Mi.bin'
        command=rc+['copyto',str(fixture),remote,'--drive-upload-cutoff','128Mi','--drive-chunk-size',f'{chunk}Mi']
        row={'chunk_mib':chunk,'remote':remote,'allowed_timeout_s':transfer_timeout(command,180)}
        started=time.time()
        try:
            checked(command,timeout=180)
            row['upload_s']=time.time()-started
            output=checked(rc+['md5sum',remote],timeout=60).stdout
            lines=re.findall(r'^([0-9a-fA-F]{32})\s',output,re.MULTILINE)
            assert len(lines)==1,'Cannot identify the checksum line in native output'
            checksum=lines[0].lower()
            assert checksum==report['fixture_md5'],'Remote MD5 differs from uploaded fixture'
            row.update(status='passed',remote_md5_verified=True)
        except Exception as error:
            row.update(status='error',elapsed_s=time.time()-started,error=sanitize(str(error)))
        report['trials'].append(row);save()
    successes=[r for r in report['trials'] if r['status']=='passed']
    assert successes,'No chunked-upload candidate passed; do not resume a GPU job yet'
    report.update(status='passed',selected_chunk_mib=min(successes,key=lambda r:r['upload_s'])['chunk_mib'],elapsed_s=time.time()-begin)
except Exception as error:
    report.update(status='error',error=sanitize(str(error)),elapsed_s=time.time()-begin)
finally:
    save()
print(json.dumps(report,indent=2),flush=True)
