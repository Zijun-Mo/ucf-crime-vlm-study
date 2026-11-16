"""Exercise the real Drive publication path with tiny synthetic files, no GPU."""
import argparse,datetime,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import configure_drive_environment,checked
from openjev_dss.durable_publish import IncrementalPublisher,remote_inventory
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'research/localization_v2/publication_live_test.json')
args=parser.parse_args()
configure_drive_environment(ROOT)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
remote='gdrive:remaining_20261108_v1/publication_smoke_'+stamp
rc=['rclone','--drive-root-folder-id','YOUR_DRIVE_ROOT_FOLDER_ID','--contimeout','15s','--timeout','60s','--retries','2']
checked(rc+['mkdir',remote],timeout=90)
with tempfile.TemporaryDirectory(prefix='publication-live-') as scratch:
 root=Path(scratch)
 for i in range(34):(root/f'{i:02d}.json').write_text(json.dumps({'fixture':i}))
 (root/'resume_probe.bin').write_bytes(b'old!');(root/'status.json').write_text('{"step":1}')
 calls=[];fail=[True];commits=[]
 def send(names,timeout):
  if len(calls)==1 and fail[0]:raise TimeoutError('Injected before second payload batch')
  listing=root.parent/(root.name+'_files.txt');listing.write_text('\n'.join(names)+'\n')
  try:checked(rc+['copy',str(root),remote,'--files-from',str(listing),'--checksum','--no-traverse','--transfers','4'],timeout=timeout)
  finally:listing.unlink(missing_ok=True)
  calls.append(names)
 def commit():
  checked(rc+['copyto',str(root/'status.json'),remote+'/status.json','--ignore-times'],timeout=120);commits.append(1)
 def inventory():return remote_inventory(rc+['lsjson',remote,'--recursive','--files-only','--hash'])
 publisher=IncrementalPublisher(root,send,commit,inventory())
 try:publisher()
 except TimeoutError:pass
 else:raise AssertionError('Injected fault was not exercised')
 partial=inventory();assert len(partial)==16 and 'status.json' not in partial and not commits
 fail[0]=False;publisher();complete=inventory()
 assert len(complete)==36 and len(commits)==1
 assert all(complete[p.name]==hashlib.md5(p.read_bytes()).hexdigest() for p in root.iterdir())
 before=len(calls);restarted=IncrementalPublisher(root,send,commit,complete);restarted()
 assert len(calls)==before
 path=root/'resume_probe.bin';stat=path.stat();path.write_bytes(b'new!');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
 restarted();assert calls[-1]==['resume_probe.bin']
 assert inventory()['resume_probe.bin']==hashlib.md5(b'new!').hexdigest()
 report=dict(passed=True,remote=remote,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_allocated=False,
  tests=['Real rclone JSON stdout despite stderr notices','Partial payload leaves status uncommitted','Retry completes remaining bounded batches','Fresh publisher skips acknowledged content','Equal-size/timestamp checkpoint changes upload correctly'],
  max_batch_files=max(map(len,calls)),payload_batches=len(calls),status_commits=len(commits))
 report['tested_sources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/openjev_dss/durable_publish.py','research/run_localization_training.py']}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False),flush=True)
