"""Exact aligned vision batches, shared immutable shards and bounded window views."""
from collections import OrderedDict
import hashlib
import json
import pickle
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from .data import sample_starts
from .runtime import decode_canonical_frames
from .study_pair_visuals import GatedPairVisualStore, _pairs

PAIR_BATCH=16
SHARD_GROUPS=64


def pair_batches(path, valid, count, groups, guard=lambda:None):
    """Yield the SAME contiguous, padded 16-pair inputs as the full producer.

    Skipped batches do not change the position or content of any selected batch.
    Decode sequentially and retain at most 32 RGB frames, including padding.
    """
    groups=sorted(set(groups));assert groups and groups[0]>=0 and groups[-1]*16<=count
    cap=cv2.VideoCapture(str(path));assert cap.isOpened();cursor=0;carry={}
    try:
        for group in groups:
            guard();first=group*16;last=min(count,first+15)
            lo=min(2*first,valid);hi=valid if last==count else min(2*last+1,valid)
            frames={i:f for i,f in carry.items() if i>=lo}
            while cursor<=hi:
                if cursor%128==0:guard()
                ok,frame=cap.read();assert ok,'Canonical ended before its frozen padding frame'
                if cursor>=lo:frames[cursor]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                cursor+=1
            selected=[]
            for pair in range(first,last+1):
                indices=[valid,valid] if pair==count else [min(2*pair+j,valid) for j in range(2)]
                selected.append(np.stack([frames[i] for i in indices]))
            actual=len(selected);carry={hi:frames[hi]}
            yield group,actual,selected+[selected[-1]]*(16-actual)
        if groups[-1]==count//16:
            assert cursor==valid+1 and not cap.read()[0],'Canonical frame count differs from metadata'
    finally:cap.release()


class TrainingMetadata:
    def __init__(self,root,records):
        self.root=Path(root);self.records={r['name']:r for r in records};self.loaded={}
    def __getitem__(self,name):
        if name not in self.loaded:
            m=json.loads((self.root/'metadata'/(name+'.json')).read_text())
            assert m['sha256']==self.records[name]['sha256']
            self.loaded[name]={**m,'source_sha256':self.records[name]['sha256']}
        return self.loaded[name]


def stream_input_gate(canonical, metadata, fixtures, guard=lambda:None):
    """Compare actual RGB arrays to the unchanged full decoder on train data."""
    checks=[]
    for record in fixtures:
        guard();m=metadata[record['name']];n=m['window_count'];valid=m['sampled_frames']
        frames=decode_canonical_frames(Path(canonical)/record['name'],valid,[0],True)
        full=_pairs(frames,valid,n);groups=sorted({0,(n//16)//2,n//16})
        for group,actual,inputs in pair_batches(Path(canonical)/record['name'],valid,n,groups,guard):
            expected=full[group*16:group*16+16];expected+= [expected[-1]]*(16-len(expected))
            equal=all(np.array_equal(a,b) for a,b in zip(inputs,expected))
            checks.append({'name':record['name'],'group':group,'actual_pairs':actual,'physical_pairs':16,'bitwise_equal':equal})
        del full,frames
    return {'allowed':bool(checks) and all(r['bitwise_equal'] for r in checks),'checks':checks,
            'gate':'Byte-identical RGB inputs; original aligned batch positions and tail padding; train fixtures only'}


class WindowView:
    """Materialize only requested windows; keep at most two <=1024-pair shards."""
    def __init__(self,provider,signature,index,count):
        self.provider=provider;self.signature=signature;self.index=index;self.count=count;self.open=OrderedDict();self.checked=set()
    def __len__(self):return self.count
    def _pair(self,pair):
        group=pair//16;path,offset,actual=self.index[group]
        if path not in self.open:
            entry=self.provider._load(path,self.signature,finite=False)
            self.open[path]=entry['features']
            while len(self.open)>2:self.open.popitem(last=False)
        self.open.move_to_end(path)
        row=pair%16;assert row<actual
        if group not in self.checked:
            assert torch.isfinite(self.open[path][offset:offset+actual]).all(),'Nonfinite generated cache group'
            self.checked.add(group)
        return self.open[path][offset+row]
    def select(self,starts):
        assert starts and all(0<=s<self.count for s in starts)
        return torch.stack([torch.cat([self._pair(min(s+j,self.count)) for j in range(4)]) for s in starts])
    def __getitem__(self,key):
        if isinstance(key,int):
            i=key if key>=0 else self.count+key
            return self.select([i])[0]
        if isinstance(key,slice):
            starts=list(range(*key.indices(self.count)));assert starts
            return self.select(starts)
        return self.select([int(i) for i in key])


class FormalVisualProvider:
    def __init__(self,runtime,data,records,local,revision,guard=lambda:None):
        self.guard=guard;self.pending=set();self.local=Path(local);self.local.mkdir(exist_ok=True)
        self.canonical=Path(data)/'canonical';self.metadata=TrainingMetadata(data,records)
        self.store=GatedPairVisualStore(runtime,self.canonical,self.metadata,local,revision,guard=guard)
        self.repairs=[];self.headers={};self.generated_groups=0;self.generated_pairs=0;self.on_shard=lambda:None
        self.expected_dtype=torch.bfloat16 if str(getattr(runtime,'device','cpu')).startswith('cuda') else None
        fixtures=[min((r for r in records if r['split']=='train' and r['label']==y),key=lambda r:(r['size'],r['name'])) for y in [0,1]]
        self.gate=self.store.gate(fixtures,4)
        assert self.gate['allowed'],'Frozen temporal-pair gate failed'
        streamed=[]
        for label in [0,1]:
            eligible=sorted([r for r in records if r['split']=='train' and r['label']==label and 32<=self.metadata[r['name']]['window_count']<=600],key=lambda r:(r['size'],r['name']))
            if not eligible:eligible=[fixtures[label]]
            for q in [0.,.25,.5,.75]:
                r=eligible[min(len(eligible)-1,int(q*len(eligible)))]
                if r not in streamed:streamed.append(r)
        self.gate['streaming_inputs']=stream_input_gate(self.canonical,self.metadata,streamed,guard)
        assert self.gate['streaming_inputs']['allowed'],'Streaming changed the original physical vision inputs'

    def _signature(self,record):
        m=self.metadata[record['name']]
        return {**self.store.signature,'representation':'immutable_aligned_pair_shards_v1','pair_batch':16,'duration_gate':4,
                'canonical_sha256':m['canonical_sha256'],'source_sha256':record['sha256'],'window_count':m['window_count']}

    def _prefix(self,signature):
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        folder=self.local/key[:2];folder.mkdir(exist_ok=True)
        return folder,key

    def cache_paths(self,record):
        folder,key=self._prefix(self._signature(record))
        return sorted(folder.glob(key+'_*.pt'))

    def cache_path(self,record):
        """First immutable shard, for diagnostics and single-shard fixtures."""
        return self.cache_paths(record)[0]

    def _load(self,path,signature,finite=False):
        entry=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert entry['signature']==signature
        groups=entry['groups'];features=entry['features'];n=signature['window_count']
        assert groups and groups==sorted(set(groups)) and all(isinstance(g,int) and 0<=g*16<=n for g in groups)
        assert len(groups)<=SHARD_GROUPS
        rows=sum(min(16,n+1-g*16) for g in groups)
        assert features.ndim==3 and features.shape[:2]==(rows,49)
        assert str(features.dtype)==entry['feature_dtype']
        if self.expected_dtype is not None:assert features.dtype==self.expected_dtype
        if finite:assert torch.isfinite(features).all()
        return entry

    def _scan(self,record,signature):
        index={}
        for path in self.cache_paths(record):
            try:
                stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
                old=self.headers.get(path)
                if old is None or old[0]!=stamp:
                    entry=self._load(path,signature,finite=True)
                    header=(entry['groups'],entry['features'].shape[2]);self.headers[path]=(stamp,header)
                else:header=old[1]
                offset=0
                for group in header[0]:
                    actual=min(16,signature['window_count']+1-group*16)
                    assert group not in index,'Overlapping immutable cache shards'
                    index[group]=(path,offset,actual);offset+=actual
            except (EOFError,RuntimeError,AssertionError,KeyError,ValueError,pickle.UnpicklingError) as error:
                quarantine=path.with_suffix('.invalid_'+str(time.time_ns()));path.rename(quarantine)
                self.pending.discard(path.relative_to(self.local).as_posix());self.headers.pop(path,None)
                self.repairs.append({'file':path.relative_to(self.local).as_posix(),'reason':type(error).__name__,
                                     'preserved_as':quarantine.relative_to(self.local).as_posix()})
                # Rebuild after quarantining so no index points into that shard.
                return self._scan(record,signature)
        return index

    def _save(self,signature,groups,values):
        folder,key=self._prefix(signature)
        shard=hashlib.sha256(json.dumps(groups).encode()).hexdigest()
        path=folder/(key+'_'+shard+'.pt');assert not path.exists()
        features=torch.cat(values).contiguous();assert torch.isfinite(features).all()
        temporary=path.with_suffix('.partial')
        torch.save({'signature':signature,'groups':groups,'feature_dtype':str(features.dtype),'features':features},temporary)
        temporary.replace(path);self.pending.add(path.relative_to(self.local).as_posix())
        self.generated_groups+=len(groups);self.generated_pairs+=len(features)
        self.on_shard()

    def __call__(self,record,epoch,exhaustive=False):
        self.guard();m=self.metadata[record['name']];n=m['window_count'];signature=self._signature(record)
        starts=None if exhaustive else sample_starts(n,epoch,record['name'])
        need=set(range(n//16+1)) if exhaustive else {min(t+j,n)//16 for t in starts for j in range(4)}
        index=self._scan(record,signature);missing=sorted(need-index.keys())
        if missing:
            source=self.canonical/record['name']
            with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['canonical_sha256']
            groups=[];values=[]
            with torch.no_grad():
                for group,actual,inputs in pair_batches(source,m['sampled_frames'],n,missing,self.guard):
                    f=self.store.runtime.vision(inputs)[:actual].detach().cpu()
                    assert f.shape[0]==actual and f.shape[1]==49 and torch.isfinite(f).all()
                    groups.append(group);values.append(f)
                    if len(groups)==SHARD_GROUPS:
                        self._save(signature,groups,values);groups=[];values=[]
                if groups:self._save(signature,groups,values)
            index=self._scan(record,signature)
        assert need<=index.keys()
        view=WindowView(self,signature,index,n)
        return view if exhaustive else view.select(starts)

    def flush(self,transfer,remote,deadline,reserve=240):
        if not self.pending:return {'status':'nothing_pending'}
        names=sorted(self.pending);size=sum((self.local/name).stat().st_size for name in names)
        estimate=min(1200,max(120,120+size/(4*1024**2)))
        if time.time()+estimate+reserve>=deadline:return {'status':'deferred_for_checkpoint_budget','files':len(names),'bytes':size}
        listing=self.local/'pending_uploads.txt';listing.write_text('\n'.join(names)+'\n')
        transfer(['copy',str(self.local),remote,'--files-from',str(listing),'--transfers','4','--checkers','4'],estimate)
        self.pending.difference_update(names)
        return {'status':'uploaded','files':len(names),'bytes':size}
