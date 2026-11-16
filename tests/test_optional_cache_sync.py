import json
import os
from pathlib import Path
import sys
import time
import pytest
from openjev_dss.optional_cache_sync import OptionalCacheSync


def native(tmp_path, mode):
    script=tmp_path/'native.py'
    script.write_text('''import hashlib,json,os,pathlib,shutil,sys,time
args=sys.argv[1:];mode=args.pop(0);root=pathlib.Path(args.pop(0))
if 'copyto' in args:
 index=args.index('copyto');source=pathlib.Path(args[index+1]);dest=pathlib.Path(args[index+2])
 (root/'child.pid').write_text(str(os.getpid()))
 if mode=='slow':time.sleep(30)
 else:time.sleep(.25)
 if mode=='fail':print('HTTP403 https://runtime.example/file?token=private123',flush=True);raise SystemExit(2)
 dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
else:
 p=pathlib.Path(args[args.index('lsjson')+1]);print(json.dumps({'Size':p.stat().st_size,'Hashes':{'md5':hashlib.md5(p.read_bytes()).hexdigest()},'ID':'verified-test-object'}))
''')
    return [sys.executable,str(script),mode,str(tmp_path)]


def test_optional_upload_overlaps_scoring_and_verifies_cloud_bytes(tmp_path):
    source=tmp_path/'cache.pt';source.write_bytes(b'immutable features')
    sync=OptionalCacheSync(native(tmp_path,'ok'),str(tmp_path/'cloud'),tmp_path/'receipt.json',time.time()+300)
    start=time.monotonic();sync.submit(source)
    assert time.monotonic()-start<.2
    # The caller can publish scores while native feature transfer is running.
    (tmp_path/'scores.json').write_text('{"score":1}')
    sync.close(grace=3)
    report=json.loads((tmp_path/'receipt.json').read_text())
    assert report['status']=='complete' and report['entries'][0]['status']=='verified'
    assert (tmp_path/'cloud/cache.pt').read_bytes()==source.read_bytes()


def test_cache_failure_is_recorded_without_aborting_the_scoring_caller(tmp_path):
    source=tmp_path/'cache.pt';source.write_bytes(b'features')
    sync=OptionalCacheSync(native(tmp_path,'fail'),str(tmp_path/'cloud'),tmp_path/'receipt.json',time.time()+300)
    sync.submit(source);sync.close(grace=3)
    report=(tmp_path/'receipt.json').read_text();entry=json.loads(report)['entries'][0]
    assert entry['status']=='error' and '403' in entry['error'] and 'private123' not in report
    assert json.loads(report)['status']=='partial'


def test_bounded_shutdown_terminates_real_native_child_and_defers_remaining_files(tmp_path):
    source=tmp_path/'cache.pt';source.write_bytes(b'features')
    sync=OptionalCacheSync(native(tmp_path,'slow'),str(tmp_path/'cloud'),tmp_path/'receipt.json',time.time()+300)
    sync.submit(source)
    until=time.monotonic()+3
    while not (tmp_path/'child.pid').exists() and time.monotonic()<until:time.sleep(.02)
    pid=int((tmp_path/'child.pid').read_text());start=time.monotonic();sync.close(grace=.1)
    assert time.monotonic()-start<4
    with pytest.raises(ProcessLookupError):os.kill(pid,0)
    report=json.loads((tmp_path/'receipt.json').read_text())
    assert report['entries'][0]['status']=='deferred' and report['pending_bytes']==0


def test_bounded_backlog_does_not_start_unaffordable_optional_work(tmp_path):
    source=tmp_path/'cache.pt';source.write_bytes(b'features')
    sync=OptionalCacheSync(['does-not-exist'],str(tmp_path/'cloud'),tmp_path/'receipt.json',time.time()+300,max_pending_bytes=0)
    sync.submit(source);sync.close(grace=1)
    report=json.loads((tmp_path/'receipt.json').read_text())
    assert report['entries'][0]['status']=='deferred' and report['pending_bytes']==0
