"""Bounded background publication of optional, immutable inference features.

Scores/checkpoints use their original synchronous publication path. A failed
feature upload is recorded for recomputation, and never certifies a cloud hit.
"""
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from .cloud_io import sanitize


class OptionalCacheSync:
    def __init__(self, command, remote, receipt, deadline, max_pending_bytes=16*1024**3):
        self.command=list(command);self.remote=remote.rstrip('/');self.receipt=Path(receipt)
        self.deadline=deadline;self.max_pending_bytes=max_pending_bytes
        self.lock=threading.Lock();self.stop=threading.Event();self.jobs=queue.Queue()
        self.rows={};self.pending_bytes=0;self.closing=False;self.closed=False
        self.thread=threading.Thread(target=self._worker,daemon=True);self.thread.start()

    def _save(self):
        data={'status':('complete' if all(r['status']=='verified' for r in self.rows.values()) else 'partial') if self.closed else 'running',
              'updated':time.time(),'optional_features_only':True,'scores_and_weights_synchronous':True,
              'pending_bytes':self.pending_bytes,'entries':list(self.rows.values())}
        self.receipt.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.receipt.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(self.receipt)

    def submit(self, path):
        path=Path(path);assert path.is_file() and path.suffix=='.pt'
        with self.lock:
            assert not self.closing
            key=str(path.resolve())
            if key in self.rows:return
            size=path.stat().st_size
            row={'source':key,'remote':self.remote+'/'+path.name,'bytes':size,'status':'queued'}
            self.rows[key]=row
            if self.pending_bytes+size>self.max_pending_bytes or shutil.disk_usage(path.parent).free<32*1024**3:
                row.update(status='deferred',reason='Optional publication backlog/disk reserve; recompute if absent on Drive')
            else:
                self.pending_bytes+=size;self.jobs.put((path,row))
            self._save()

    def _run(self, command, limit):
        remaining=min(limit,self.deadline-time.time()-90)
        if remaining<=0:raise TimeoutError('Optional feature publication deadline reserve')
        end=time.monotonic()+remaining
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            while True:
                if self.stop.is_set() or time.monotonic()>=end:
                    raise TimeoutError('Optional feature upload canceled at its bounded publication limit')
                try:
                    output=process.communicate(timeout=min(.25,max(.01,end-time.monotonic())))[0]
                    break
                except subprocess.TimeoutExpired:pass
            if process.returncode:raise RuntimeError(f'Native transfer exit={process.returncode}: '+sanitize(output)[-3000:])
            return output
        finally:
            if process.poll() is None:
                process.terminate()
                try:output=process.communicate(timeout=3)[0]
                except subprocess.TimeoutExpired:
                    process.kill();output=process.communicate(timeout=3)[0]
                with self.lock:
                    self.last_interrupted_output=sanitize(output)[-3000:]

    def _worker(self):
        while True:
            try:path,row=self.jobs.get(timeout=.1)
            except queue.Empty:
                if self.closing:return
                continue
            try:
                if self.stop.is_set():raise TimeoutError('Optional feature publication deferred at worker shutdown')
                with self.lock:
                    self.last_interrupted_output='';row.update(status='uploading',started=time.time());self._save()
                stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
                with path.open('rb') as stream:md5=hashlib.file_digest(stream,'md5').hexdigest()
                self._run(self.command+['copyto',str(path),row['remote']],1200)
                server=json.loads(self._run(self.command+['lsjson',row['remote'],'--stat','--hash','--log-level','ERROR'],45))
                assert server['Size']==row['bytes'] and server['Hashes']['md5']==md5
                stat=path.stat();assert stamp==(stat.st_size,stat.st_mtime_ns),'Immutable feature file changed during upload'
                with self.lock:row.update(status='verified',md5=md5,server_id=server.get('ID'))
            except Exception as error:
                with self.lock:
                    row.update(status='deferred' if isinstance(error,TimeoutError) else 'error',
                               error=sanitize(str(error)),last_output=getattr(self,'last_interrupted_output',''))
            finally:
                with self.lock:
                    row['finished']=time.time();self.pending_bytes-=row['bytes'];self._save()
                self.jobs.task_done()

    def close(self, grace=120):
        with self.lock:self.closing=True
        self.thread.join(timeout=max(0,min(grace,self.deadline-time.time()-90)))
        if self.thread.is_alive():self.stop.set();self.thread.join(timeout=10)
        assert not self.thread.is_alive(),'Optional cache publisher did not terminate its native child'
        with self.lock:self.closed=True;self._save()
