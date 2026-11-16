"""Bounded local scratch for immutable, Drive-persisted visual features."""
import hashlib,shutil,threading,uuid
from collections import OrderedDict
from pathlib import Path
import torch

class FeatureCache:
    def __init__(self,local,memory_bytes=1024**3,disk_bytes=96*1024**3):
        self.local=Path(local);self.local.mkdir(parents=True,exist_ok=True)
        self.memory_limit=memory_bytes;self.disk_limit=disk_bytes
        self.memory=OrderedDict();self.memory_used=0;self.lock=threading.RLock()
        self.stats={'memory_hits':0,'disk_hits':0,'drive_reads':0,'bytes_from_drive':0}
    def local_path(self,path):
        return self.local/(hashlib.sha256(str(path).encode()).hexdigest()+'.pt')
    def remember(self,key,signature,value):
        size=value.numel()*value.element_size()
        if size>self.memory_limit:return
        if key in self.memory:
            _,old=self.memory.pop(key);self.memory_used-=old.numel()*old.element_size()
        self.memory[key]=(signature,value);self.memory_used+=size
        while self.memory_used>self.memory_limit:
            _,(_,old)=self.memory.popitem(last=False);self.memory_used-=old.numel()*old.element_size()
    def trim_disk(self,keep):
        files=sorted(self.local.glob('*.pt'),key=lambda p:p.stat().st_mtime)
        total=sum(p.stat().st_size for p in files)
        for path in files:
            if total<=self.disk_limit:break
            if path!=keep:total-=path.stat().st_size;path.unlink()
    def get(self,path,signature):
        key=str(path)
        with self.lock:
            if key in self.memory:
                expected,value=self.memory.pop(key);self.memory[key]=(expected,value)
                assert expected==signature,'Visual cache fingerprint mismatch'
                self.stats['memory_hits']+=1;return value
            local=self.local_path(path)
            if local.exists():self.stats['disk_hits']+=1
            elif Path(path).exists():
                tmp=local.with_suffix('.tmp-'+uuid.uuid4().hex)
                try:shutil.copyfile(path,tmp);tmp.replace(local)
                finally:tmp.unlink(missing_ok=True)
                self.stats['drive_reads']+=1;self.stats['bytes_from_drive']+=local.stat().st_size
            else:return None
            cached=torch.load(local,map_location='cpu',weights_only=True,mmap=True)
            assert cached['signature']==signature,'Visual cache fingerprint mismatch'
            value=cached['features'];self.remember(key,signature,value);self.trim_disk(local);return value
    def put(self,path,signature,value):
        with self.lock:
            path=Path(path);local=self.local_path(path);tmp=local.with_suffix('.tmp-'+uuid.uuid4().hex)
            torch.save({'signature':signature,'features':value.detach().cpu()},tmp);tmp.replace(local)
            # Drive is authoritative. Local scratch is only a disposable replica.
            persistent_tmp=path.with_suffix('.tmp-'+uuid.uuid4().hex)
            try:shutil.copyfile(local,persistent_tmp);persistent_tmp.replace(path)
            finally:persistent_tmp.unlink(missing_ok=True)
            self.remember(str(path),signature,value);self.trim_disk(local)
