import hashlib
from pathlib import Path
import pytest
from openjev_dss.durable_publish import IncrementalPublisher,remote_inventory


def test_partial_upload_never_advances_pointer_and_retry_skips_confirmed_chunks(tmp_path):
    for name in ['a.json','b.json','c.json','status.json']:(tmp_path/name).write_text(name)
    calls=[];fail=[True];commits=[]
    def send(names,seconds):
        calls.append(names)
        if names==['c.json'] and fail[0]:raise TimeoutError('Injected partial publication')
    pub=IncrementalPublisher(tmp_path,send,lambda:commits.append('status'),chunk_size=2)
    with pytest.raises(TimeoutError):pub()
    assert calls==[['a.json','b.json'],['c.json']] and not commits
    fail[0]=False;pub()
    assert calls[-1]==['c.json'] and commits==['status']
    calls.clear();pub();assert calls==[] and commits==['status','status']


def test_restart_uses_content_hash_and_detects_equal_size_checkpoint_changes(tmp_path):
    path=tmp_path/'resume_0.pt';path.write_bytes(b'old!');(tmp_path/'status.json').write_text('{}')
    remote={'resume_0.pt':hashlib.md5(b'old!').hexdigest()};sent=[];commits=[]
    pub=IncrementalPublisher(tmp_path,lambda names,seconds:sent.append(names),lambda:commits.append(1),remote)
    pub();assert not sent
    import os
    stamp=path.stat();path.write_bytes(b'new!');os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    pub();assert sent==[['resume_0.pt']] and len(commits)==2


def test_no_single_publication_batch_contains_thousands_of_files(tmp_path):
    for i in range(1001):(tmp_path/f'{i:04d}.json').write_text('{}')
    (tmp_path/'status.json').write_text('{}');calls=[]
    pub=IncrementalPublisher(tmp_path,lambda names,seconds:calls.append((names,seconds)),lambda:None)
    pub();assert len(calls)==63 and max(len(names) for names,_ in calls)==16
    assert all(seconds<=600 for _,seconds in calls)


def test_inventory_does_not_parse_stderr_notices_as_json(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    def run(command,**kwargs):
        assert kwargs['stderr']==subprocess.PIPE and kwargs['stdout']==subprocess.PIPE
        return SimpleNamespace(returncode=0,stdout='[{"Path":"a","Hashes":{"md5":"hash"}}]',
            stderr='2026/11/12 NOTICE: shared client is being retired\n')
    monkeypatch.setattr(subprocess,'run',run)
    assert remote_inventory(['rclone','lsjson'])=={'a':'hash'}
