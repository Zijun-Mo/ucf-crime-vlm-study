import copy
import itertools
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from openjev_dss.localization_objective import selection
from openjev_dss.localization_pointwise import coefficients,plan,terms,backward_pointwise,window_budget


def test_window_loss_and_gradient_do_not_depend_on_other_sampled_windows():
    z=torch.tensor([1.,-2.,4.],requires_grad=True)
    value=terms(z,1,[.13,.42,.1],[.07,.2,.05],[2.,-3.,5.],2.)
    grad=torch.autograd.grad(value.sum(),z)[0]
    single=torch.tensor([1.],requires_grad=True)
    own=terms(single,1,[.13],[.07],[2.],2.)
    torch.testing.assert_close(value[:1],own)
    torch.testing.assert_close(grad[:1],torch.autograd.grad(own.sum(),single)[0])
    other=torch.tensor([1.,200.,-100.],requires_grad=True)
    changed=terms(other,1,[.13,.99,0.],[.07,2.,4.],[2.,30.,50.],2.)
    torch.testing.assert_close(value[:1],changed[:1])
    torch.testing.assert_close(grad[:1],torch.autograd.grad(changed.sum(),other)[0][:1])


@pytest.mark.parametrize('label',[0,1])
def test_sampling_coefficients_are_fixed_and_streaming_matches_dense(label):
    torch.manual_seed(15);model=torch.nn.Linear(2,1);stream=copy.deepcopy(model)
    x=torch.randn(37,2);chosen=selection([-3.]*19+[4.]*18,dict(threshold=0.,scale=1.))
    config=dict(fraction=.25,normal_cap=8,positive_cap=5,anchor_cap=6)
    sampled=plan('video',label,len(x),chosen,2,config)
    assert sampled==plan('video',label,len(x),chosen,2,config)
    indices=sampled['indices'];z=torch.cat([model(x[indices[i:i+4]]).flatten() for i in range(0,len(indices),4)])
    loss=terms(z,label,sampled['bce'],sampled['anchor'],sampled['teacher'],sampled['scale']).sum()
    (loss*.3).backward()
    value,info=backward_pointwise(stream,x,sampled,scale=.3)
    assert value==pytest.approx(float(loss.detach()),rel=1e-6)
    assert info['window_independent'] and info['score_calls']==0
    for a,b in zip(model.parameters(),stream.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
    if label:
        a=coefficients(chosen);p=min(5,int((a>0).sum()))/int((a>0).sum())
        for i,v in zip(indices,sampled['bce']):
            if v:assert v==pytest.approx(a[i]/p)


def test_empty_positive_selection_never_creates_positive_reward():
    selected=selection([-5.]*20,dict(threshold=0.,scale=1.))
    sampled=plan('empty',1,20,selected,0,dict(fraction=.25,normal_cap=8,positive_cap=5,anchor_cap=6))
    assert not any(sampled['bce']) and sampled['anchor_draws']==5
    z=torch.full((5,),-5.,requires_grad=True)
    loss=terms(z,1,sampled['bce'],sampled['anchor'],sampled['teacher'],sampled['scale']).sum()
    loss.backward();assert float(loss.detach())==0. and not z.grad.any()


def test_uniform_importance_weighting_recovers_full_fixed_objective_in_expectation():
    # Exhaustively integrate every possible uniform subset, avoiding a flaky
    # Monte Carlo approximation when checking the estimator's normalization.
    z=torch.tensor([-2.,.5,3.,4.]);a=torch.tensor([0.,0.,.2,.4]);t=torch.tensor([-3.,-1.,4.,5.])
    expected=terms(z,1,a,[.25]*4,t,1.).sum()
    estimates=[]
    for positive in itertools.combinations([2,3],1):
        for anchor in itertools.combinations(range(4),2):
            b=torch.zeros(4);r=torch.zeros(4);b[list(positive)]=a[list(positive)]*2;r[list(anchor)]=.5
            estimates.append(terms(z,1,b,r,t,1.).sum())
    torch.testing.assert_close(torch.stack(estimates).mean(),expected)


def test_budget_never_reads_test_metadata_and_bounds_union_and_padding():
    records=[dict(name='n',label=0,split='train'),dict(name='a',label=1,split='train'),dict(name='v',label=0,split='val'),dict(name='test',label=1,split='test')]
    meta={k:dict(window_count=n) for k,n in [('n',1000),('a',1000),('v',5)]}
    b=window_budget(records,meta,dict(fraction=.25,normal_cap=64,positive_cap=32,anchor_cap=32))
    assert b['training_physical_windows_upper']==128 and b['validation_physical_windows']==8


def test_teacher_import_preserves_logits_and_rejects_identity_change(tmp_path):
    import json
    from openjev_dss.localization_teacher_import import import_teacher,fingerprint
    from openjev_dss.learning import atomic_json
    manifest=dict(records=[dict(name='a',label=1,split='train',sha256='sha')])
    policy=dict(backbone_revision='pin',train_manifest_sha256='manifest',training_metadata_bundle_sha256='metadata',
        teacher=dict(frozen=True),input=dict(prompt='P1',duration=4,stride=1,batch=4),student='old')
    new=copy.deepcopy(policy);new['student']='pointwise'
    source=tmp_path/'old';source.mkdir();target=tmp_path/'new'
    row=dict(name='a',source_sha256='sha',fingerprint=fingerprint(manifest,policy),windows=3,
        logits=[-2.,3.,4.],teacher='pinned_zero_shot_adapter_disabled')
    atomic_json(source/'a.json',row)
    report=import_teacher(source,target,manifest,{'a':dict(window_count=3)},policy,new)
    result=json.loads((target/'a.json').read_text())
    assert result['logits']==row['logits'] and result['fingerprint']==fingerprint(manifest,new)
    assert report['imported_videos']==1 and result['imported_from']['values_unchanged']
    changed=copy.deepcopy(new);changed['input']['duration']=8
    with pytest.raises(AssertionError):import_teacher(source,target,manifest,{'a':dict(window_count=3)},policy,changed)
