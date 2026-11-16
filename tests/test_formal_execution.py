import json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch
from openjev_dss import formal_execution
from openjev_dss.formal_preflight import gradient_check


class Language(torch.nn.Linear):
    def __init__(self):super().__init__(1,1);self.checkpointing=True
    def gradient_checkpointing_disable(self):self.checkpointing=False


def runtime(name):
    model=Language()
    return SimpleNamespace(name=name,model=model,head=torch.nn.Identity(),device='cpu',forward=lambda x:model(x).flatten())


def test_dense_repeat_external_gradient_and_leaf_pass_agree_on_a_real_autograd_graph(tmp_path):
    rt=runtime('qwen');bags=[torch.tensor([[1.],[2.]]),torch.tensor([[3.],[4.]])]
    report=gradient_check(rt,bags,tmp_path/'diagnostic.json')
    assert report['passed'] and report['dense_repeat']['relative_l2']==0
    assert report['direct_external_gradient']['relative_l2']==0
    assert report['dense_repeat']['upstream_max_abs']==0


def test_execution_requires_both_arms_and_frozen_choice_cannot_fall_back_on_resume(tmp_path,monkeypatch):
    runtimes={n:runtime(n) for n in ['qwen','openjev']};calls=[]
    def gate(rt,bags,path,strict):
        mode=path.name.split('_')[0];calls.append((mode,rt.name))
        return {'passed':mode=='math' or rt.name=='qwen'}
    monkeypatch.setattr(formal_execution,'gradient_check',gate)
    choice,results=formal_execution.select_execution(runtimes,[],tmp_path)
    assert choice=={'attention_backend':'math','gradient_checkpointing':False}
    assert calls==[('auto','qwen'),('auto','openjev'),('math','qwen'),('math','openjev')]
    frozen=(tmp_path/'training_execution.json').read_bytes();calls.clear()
    monkeypatch.setattr(formal_execution,'gradient_check',lambda *args,**kwargs:{'passed':False})
    with pytest.raises(RuntimeError,match='No common formal execution'):
        formal_execution.select_execution(runtimes,[],tmp_path)
    assert (tmp_path/'training_execution.json').read_bytes()==frozen


def test_math_backend_is_training_only_and_validation_uses_original_forward(monkeypatch):
    import torch.nn.attention
    entries=[]
    @contextmanager
    def context(backend):entries.append(backend);yield
    monkeypatch.setattr(torch.nn.attention,'sdpa_kernel',context)
    rt=runtime('qwen');original=rt.forward;features=torch.tensor([[1.],[2.]])
    formal_execution.apply_execution({'qwen':rt},{'attention_backend':'math','gradient_checkpointing':False})
    rt.model.train();torch.testing.assert_close(rt.forward(features),original(features));assert len(entries)==1
    rt.model.eval();torch.testing.assert_close(rt.forward(features),original(features));assert len(entries)==1
