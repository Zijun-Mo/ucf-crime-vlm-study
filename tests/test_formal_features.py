import hashlib
import importlib.metadata
import json
import time
import cv2
import numpy as np
import pytest
import torch
from openjev_dss.formal_features import FormalVisualProvider


class Encoder:
    def __init__(self):self.calls=0
    def vision(self,windows):
        self.calls+=1
        return torch.stack([torch.cat([torch.full((49,2),float(w[i:i+2].astype(np.float32).mean()))
                                      for i in range(0,len(w),2)]) for w in windows])


def provider(tmp_path,monkeypatch,count=4,valid=None):
    actual=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'test' if name=='transformers' else actual(name))
    (tmp_path/'metadata').mkdir();(tmp_path/'canonical').mkdir()
    records=[];valid=2*count-1 if valid is None else valid
    for label in [0,1]:
        name=f'{label}.mp4';path=tmp_path/'canonical'/name
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2,(16,16))
        for i in range(valid+1):writer.write(np.full((16,16,3),(i*13+label)%220,np.uint8))
        writer.release()
        record={'name':name,'sha256':'source'+str(label),'n_frames':(count*30 if valid==2*count else count*30-15),'fps':30,'split':'train','label':label,'size':path.stat().st_size}
        metadata={'sha256':record['sha256'],'canonical_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sampled_frames':valid,'window_count':count}
        (tmp_path/'metadata'/(name+'.json')).write_text(json.dumps(metadata));records.append(record)
    encoder=Encoder()
    return FormalVisualProvider(encoder,tmp_path,records,tmp_path/'cache','test'),records,encoder


def test_shared_cache_reuses_features_across_epochs_and_repairs_generated_corruption(tmp_path,monkeypatch):
    p,records,encoder=provider(tmp_path,monkeypatch)
    expected=p(records[0],0);calls=encoder.calls
    assert torch.equal(expected,p(records[0],2)) and encoder.calls==calls
    path=p.cache_path(records[0]);torch.save({'signature':'corrupted'},path)
    assert torch.equal(expected,p(records[0],3))
    assert len(p.repairs)==1 and (p.local/p.repairs[0]['preserved_as']).exists()
    assert path.relative_to(p.local).as_posix() in p.pending


def test_failed_cache_transfer_keeps_pending_files_and_allocation_reserve(tmp_path,monkeypatch):
    p,records,_=provider(tmp_path,monkeypatch);p(records[0],0)
    def fail(command,timeout):raise TimeoutError('network transfer')
    with pytest.raises(TimeoutError):p.flush(fail,'gdrive:test',time.time()+1000)
    assert len(p.pending)==1
    assert p.flush(fail,'gdrive:test',time.time()+1)['status']=='deferred_for_checkpoint_budget'
    assert len(p.pending)==1
    transferred=[]
    def transfer(command,timeout):transferred.append((command,timeout))
    assert p.flush(transfer,'gdrive:test',time.time()+1000)['status']=='uploaded'
    assert not p.pending and len(transferred)==1


@pytest.mark.parametrize('count,valid',[(1,1),(1,2),(15,29),(16,31),(16,32),(17,33),(31,61),(32,63),(32,64),(33,65)])
def test_streamed_batches_are_byte_identical_to_original_full_decoder(tmp_path,monkeypatch,count,valid):
    from openjev_dss.formal_features import pair_batches
    from openjev_dss.runtime import decode_canonical_frames
    from openjev_dss.study_pair_visuals import _pairs
    p,records,_=provider(tmp_path,monkeypatch,count,valid);r=records[0];m=p.metadata[r['name']]
    frames=decode_canonical_frames(p.canonical/r['name'],valid,[0],True)
    original=_pairs(frames,valid,count)
    groups=sorted({0,count//16})
    for group,actual,batch in pair_batches(p.canonical/r['name'],valid,count,groups):
        expected=original[group*16:group*16+16]
        assert actual==len(expected) and len(batch)==16
        expected=expected+[expected[-1]]*(16-len(expected))
        assert all(np.array_equal(a,b) for a,b in zip(batch,expected))


def test_sparse_epoch_samples_and_streamed_validation_match_original_features(tmp_path,monkeypatch):
    from openjev_dss.data import sample_starts
    from openjev_dss.formal_features import WindowView
    from openjev_dss.study_pair_visuals import GatedPairVisualStore
    p,records,encoder=provider(tmp_path,monkeypatch,1056);record=records[0]
    oracle=GatedPairVisualStore(encoder,p.canonical,p.metadata,tmp_path/'oracle','test')
    oracle.pair_gate={'allowed':True};full=oracle.get(record,4)
    for epoch in range(3):
        selected=p(record,epoch)
        assert torch.equal(selected,full[torch.as_tensor(sample_starts(len(full),epoch,record['name']))])
        generated=p.generated_groups
        assert torch.equal(selected,p(record,epoch)) and p.generated_groups==generated
        if epoch==0:assert generated<len(full)//16
    view=p(record,3,True);assert isinstance(view,WindowView) and len(view)==len(full)
    for start in range(0,len(full),4):
        assert torch.equal(view[start:start+4],full[start:start+4])
        assert len(view.open)<=2
    assert p.generated_pairs==len(full)+1
    assert all(len(torch.load(path,weights_only=True)['groups'])<=64 for path in p.cache_paths(record))
    generated=p.generated_groups
    assert torch.equal(p(record,4),full[torch.as_tensor(sample_starts(len(full),4,record['name']))])
    assert p.generated_groups==generated


def test_interrupted_extraction_resumes_completed_immutable_shards(tmp_path,monkeypatch):
    p,records,encoder=provider(tmp_path,monkeypatch,1200);record=records[0]
    def stop_after_first_shard():
        if p.generated_groups>=64:raise TimeoutError('allocation boundary during long-video features')
    p.guard=stop_after_first_shard
    with pytest.raises(TimeoutError):p(record,0,True)
    paths=p.cache_paths(record);assert len(paths)==1
    committed=paths[0].read_bytes()
    p.guard=lambda:None
    view=p(record,0,True)
    assert paths[0].read_bytes()==committed
    assert p.generated_groups==1200//16+1 and p.generated_pairs==1201
    assert torch.isfinite(view[-1]).all()
