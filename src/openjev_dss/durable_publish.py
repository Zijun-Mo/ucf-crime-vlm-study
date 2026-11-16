"""Bounded, restartable payload publication with the status pointer last."""
import hashlib
import json
from pathlib import Path
import subprocess
from .cloud_io import sanitize


def remote_inventory(command,timeout=180):
    """Keep JSON stdout separate from rclone notices on stderr."""
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('Remote inventory failed: '+sanitize(result.stderr)[-2000:])
    rows=json.loads(result.stdout)
    if not isinstance(rows,list):raise ValueError('Expected rclone inventory array')
    return {r['Path']:r.get('Hashes',{}).get('md5') for r in rows}


class IncrementalPublisher:
    def __init__(self,root,send,commit_status,remote_md5=None,chunk_size=16):
        self.root=Path(root);self.send=send;self.commit_status=commit_status
        self.remote=dict(remote_md5 or {});self.chunk_size=chunk_size
        assert chunk_size>0

    def _digest(self,path):
        # Checkpoint sizes are stable, and filesystem timestamps can collide.
        # Hash actual bytes rather than risk acknowledging stale model weights.
        with path.open('rb') as inp:return hashlib.file_digest(inp,'md5').hexdigest()

    def __call__(self):
        changed=[]
        for path in sorted(self.root.rglob('*')):
            if not path.is_file() or path.suffix in ['.tmp','.partial']:continue
            key=path.relative_to(self.root).as_posix()
            if key=='status.json':continue
            md5=self._digest(path)
            if self.remote.get(key)!=md5:changed.append((key,md5,path.stat().st_size))
        for begin in range(0,len(changed),self.chunk_size):
            chunk=changed[begin:begin+self.chunk_size]
            seconds=max(120,min(600,30+8*len(chunk)+sum(r[2] for r in chunk)/(2*1024**2)))
            self.send([r[0] for r in chunk],seconds)
            # Only acknowledge confirmed chunks. A failed later chunk is retried
            # without discarding successful progress or advancing status.
            self.remote.update((key,md5) for key,md5,_ in chunk)
        if (self.root/'status.json').exists():self.commit_status()
