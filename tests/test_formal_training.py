import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from openjev_dss.formal_training import run_formal_mil
from openjev_dss.controlled_training import tensor_digest


class Model(torch.nn.Linear):
    def save_pretrained(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),Path(path)/'weights.pt')


def runtimes():
    result={}
    for name in ['qwen','openjev']:
        model=Model(2,1)
        with torch.no_grad():model.weight.fill_(.1);model.bias.zero_()
        head=torch.nn.Identity() if name=='qwen' else torch.nn.Linear(1,1)
        if name=='openjev':
            with torch.no_grad():head.weight.fill_(1.);head.bias.zero_()
        full=torch.nn.Module();full.add_module('model',model)
        full.register_parameter('frozen',torch.nn.Parameter(torch.ones(1),requires_grad=False))
        result[name]=SimpleNamespace(model=model,head=head,full=full,device='cpu',
            forward=lambda x,m=model,h=head:h(m(x)).flatten())
    return result


def inputs():
    train=[{'name':str(y),'label':y,'split':'train'} for y in [0,1]]
    val=[{'name':'val'+str(i),'label':i%2,'split':'val'} for i in range(163)]
    manifest={'records':train+val,'audit':{'near_duplicate_visual_audit':'passed'}}
    policy={'training':{'steps_per_epoch':1},'validation':{'videos':163},
            'optimizer':{'lora_lr':1e-5,'scalar_head_lr':1e-4,'weight_decay':.01,'clip_grad_norm':1.},
            'selection':{'min_delta':.002,'patience':3,'min_epochs':3}}
    def provider(r,epoch,exhaustive):
        sign=1. if r['label'] else -1.
        return torch.tensor([[sign,.1],[sign,.2]])
    return manifest,policy,provider


def final_digest(folder,name):
    state=json.loads((folder/'status.json').read_text())
    saved=torch.load(folder/name/f"resume_{state['generation']%2}.pt",map_location='cpu',weights_only=True)
    return tensor_digest(saved['trainable'])


def test_paired_training_stops_only_after_full_validation_plateau(tmp_path):
    manifest,policy,provider=inputs()
    state=run_formal_mil(runtimes(),manifest,provider,tmp_path,time.time()+1000,policy,10)
    assert state['status']=='complete' and not state['test_used']
    for name,arm in state['arms'].items():
        assert arm['steps']==4 and len(arm['validation'])==4
        assert arm['best_epoch']==1 and arm['stop_reason']=='validation_plateau'
        assert arm['frozen_parameters_unchanged']
        assert (tmp_path/arm['selected_artifact']['state_file']).exists()
        assert '1' in arm['epoch_artifacts']


def test_epoch_budget_requires_extension_and_resume_matches_direct(tmp_path):
    manifest,policy,provider=inputs()
    direct=tmp_path/'direct';resumed=tmp_path/'resumed'
    assert run_formal_mil(runtimes(),manifest,provider,direct,time.time()+1000,policy,4)['status']=='complete'
    partial=run_formal_mil(runtimes(),manifest,provider,resumed,time.time()+1000,policy,2)
    assert partial['status']=='needs_extension'
    complete=run_formal_mil(runtimes(),manifest,provider,resumed,time.time()+1000,policy,4)
    assert complete['status']=='complete'
    for name in ['qwen','openjev']:assert final_digest(direct,name)==final_digest(resumed,name)


def test_partial_validation_resumes_under_same_weights_without_retraining(tmp_path):
    manifest,policy,provider=inputs();out=tmp_path/'run';first=True
    def stop_once():
        nonlocal first
        s=json.loads((out/'status.json').read_text())
        if first and s.get('validation_progress',{}).get('qwen')==10:
            first=False;raise TimeoutError('simulated allocation boundary during validation')
    partial=run_formal_mil(runtimes(),manifest,provider,out,time.time()+1000,policy,4,stop_once)
    assert partial['status']=='budget_stopped'
    assert partial['arms']['qwen']['steps']==partial['arms']['openjev']['steps']==1
    saved=json.loads((out/'qwen/epoch_1_validation.json').read_text())
    assert len(saved['rows'])==10
    complete=run_formal_mil(runtimes(),manifest,provider,out,time.time()+1000,policy,4)
    assert complete['status']=='complete'
    assert all(arm['steps']==4 for arm in complete['arms'].values())


def test_half_completed_paired_step_resumes_each_optimizer_once(tmp_path):
    manifest,policy,provider=inputs();direct=tmp_path/'direct';out=tmp_path/'resumed'
    run_formal_mil(runtimes(),manifest,provider,direct,time.time()+1000,policy,4)
    rt=runtimes();original=rt['openjev'].forward
    def interrupted(x):raise TimeoutError('boundary after the Qwen optimizer step')
    rt['openjev'].forward=interrupted
    state=run_formal_mil(rt,manifest,provider,out,time.time()+1000,policy,4)
    assert state['status']=='budget_stopped'
    assert state['arms']['qwen']['steps']==1 and state['arms']['openjev']['steps']==0
    state=run_formal_mil(runtimes(),manifest,provider,out,time.time()+1000,policy,4)
    assert state['status']=='complete'
    for name in ['qwen','openjev']:assert final_digest(direct,name)==final_digest(out,name)


def test_two_failed_pair_publications_preserve_both_durable_optimizers(tmp_path,monkeypatch):
    import shutil
    import pytest
    from openjev_dss import formal_training as training
    manifest,policy,provider=inputs();policy['training']['steps_per_epoch']=20
    normal,abnormal=manifest['records'][:2]
    monkeypatch.setattr(training,'schedule',lambda records,epochs:[(ep,[(normal,abnormal)]*4) for ep in range(epochs) for _ in range(20)])
    local=tmp_path/'local';cloud=tmp_path/'cloud';cloud.mkdir();calls=0
    def sync():
        nonlocal calls
        calls+=1
        for path in local.glob('*/resume_*.pt'):
            target=cloud/path.relative_to(local);target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,target)
        if calls>=2:raise TimeoutError('data uploaded, then status-pointer commit failed')
        shutil.copyfile(local/'status.json',cloud/'status.json')
    with pytest.raises(TimeoutError):
        run_formal_mil(runtimes(),manifest,provider,local,time.time()+1000,policy,4,sync)
    state=json.loads((cloud/'status.json').read_text())
    assert calls==3
    for name,arm in state['arms'].items():
        saved=torch.load(cloud/name/f"resume_{state['generation']%2}.pt",map_location='cpu',weights_only=True)
        assert saved['steps']==arm['steps']==10
