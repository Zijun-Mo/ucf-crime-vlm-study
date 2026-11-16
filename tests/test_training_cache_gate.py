import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from openjev_dss import training_cache_gate as gate


def fixture(tmp_path, monkeypatch, perturb=False):
    records=[{'name':f'{y}_{i}.mp4','label':y,'split':'train','size':i,'sha256':f'source-{y}-{i}'} for y in [0,1] for i in range(8)]
    records += [{'name':'test.mp4','label':1,'split':'test','size':0,'sha256':'never-use'}]
    root=tmp_path/'data';(root/'canonical').mkdir(parents=True)
    for r in records:(root/'canonical'/r['name']).write_bytes(b'canonical')
    saved=torch.full((16,1,1),4.,dtype=torch.bfloat16)
    if perturb:saved[0,0,0]=4.03125
    def descriptor(r,epoch,exhaustive):
        return ({'sha256':r['sha256'],'canonical_sha256':hashlib.sha256(b'canonical').hexdigest(),'sampled_frames':31},list(range(16)),{'name':r['name']},Path(r['name']+'.pt'))
    provider=SimpleNamespace(root=root,descriptor=descriptor,cache=SimpleNamespace(get=lambda path,signature:saved))
    calls=[]
    class BatchSensitiveRuntime:
        def vision(self,windows):
            calls.append(len(windows))
            # Reproduces the original false-negative mechanism: a two-window
            # probe differs from stored four-window batches.
            return torch.full((len(windows),1,1),float(len(windows)),dtype=torch.bfloat16)
    monkeypatch.setattr(gate,'decode_canonical_frames',lambda *args:{i:np.zeros((2,2,3),dtype=np.uint8) for i in range(32)})
    return records,provider,BatchSensitiveRuntime(),calls


def test_identical_physical_batches_admit_all_fixed_training_fixtures(tmp_path,monkeypatch):
    records,provider,runtime,calls=fixture(tmp_path,monkeypatch)
    result=gate.verify_training_cache(runtime,provider,records,tmp_path/'old',tmp_path/'scratch')
    assert result['cache_reuse_allowed'] and len(result['checks'])==8
    assert calls==[4]*32
    assert all(row['compared_windows']==16 for row in result['checks'])
    assert 'test.mp4' not in result['fixtures']


def test_one_bfloat16_difference_blocks_reuse_without_relaxing_tolerance(tmp_path,monkeypatch):
    records,provider,runtime,calls=fixture(tmp_path,monkeypatch,perturb=True)
    result=gate.verify_training_cache(runtime,provider,records,tmp_path/'old',tmp_path/'scratch')
    assert not result['cache_reuse_allowed'] and len(calls)==1
    assert result['checks'][0]['max_abs']>0


def test_missing_cache_does_not_spend_gpu_compute(tmp_path,monkeypatch):
    records,provider,runtime,calls=fixture(tmp_path,monkeypatch)
    provider.cache.get=lambda *args:None
    assert not gate.verify_training_cache(runtime,provider,records,tmp_path/'old',tmp_path/'scratch')['cache_reuse_allowed']
    assert calls==[]


def test_changed_source_metadata_is_a_data_error(tmp_path,monkeypatch):
    records,provider,runtime,calls=fixture(tmp_path,monkeypatch)
    original=provider.descriptor
    def changed(*args):
        m,s,g,p=original(*args);m['sha256']='wrong';return m,s,g,p
    provider.descriptor=changed
    with pytest.raises(RuntimeError,match='frozen manifest'):
        gate.verify_training_cache(runtime,provider,records,tmp_path/'old',tmp_path/'scratch')


def test_short_bag_uses_the_original_unpadded_tail_batch(tmp_path,monkeypatch):
    records,provider,runtime,calls=fixture(tmp_path,monkeypatch)
    original=provider.descriptor
    def short(*args):
        m,s,g,p=original(*args);return m,s[:6],g,p
    provider.descriptor=short
    saved=torch.cat([torch.full((4,1,1),4.,dtype=torch.bfloat16),torch.full((2,1,1),2.,dtype=torch.bfloat16)])
    provider.cache.get=lambda *args:saved
    report=gate.verify_training_cache(runtime,provider,records,tmp_path/'old',tmp_path/'scratch')
    assert report['cache_reuse_allowed'] and calls==[4,2]*8
    assert all(row['compared_windows']==6 for row in report['checks'])
