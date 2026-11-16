import copy,itertools
import numpy as np
import pytest
import torch
from openjev_dss.localization_soft import targets,terms,plan,backward,window_budget


def test_full_video_probability_minmax_and_constant_abstention():
    z=[-4.,-1.,2.,5.];p=torch.tensor(z,dtype=torch.float64).sigmoid().numpy()
    row=targets(z,1)
    np.testing.assert_allclose(row['targets'],(p-p.min())/np.ptp(p))
    assert row['targets'][0]==0. and row['targets'][-1]==1.
    assert targets(z,0)['targets']==[0.]*4
    assert not targets([2.]*9,1)['active']
    assert plan('constant',1,9,targets([2.]*9,1),0,dict(fraction=.25,normal_cap=64,anomaly_cap=64))['indices']==[]
    with pytest.raises(ValueError):targets([float('nan')],1)


def test_same_window_loss_and_gradient_independent_of_companions():
    # Coefficients and targets are fixed BEFORE drawing a subset.
    full=targets([-4.,-1.,2.,5.],1)['targets'];a=.125
    for companions in [[],[-20.],[3.,99.]]:
        z=torch.tensor([.7]+companions,requires_grad=True)
        loss=terms(z,[full[1]]+[.8]*len(companions),[a]*(1+len(companions)))
        loss.sum().backward()
        assert float(loss[0])==pytest.approx(float(terms(torch.tensor([.7]),[full[1]],[a])[0]))
        assert float(z.grad[0])==pytest.approx(a*(torch.sigmoid(torch.tensor(.7)).item()-full[1]),abs=1e-8)


def test_exact_subset_expectation_matches_full_loss_and_score_gradient():
    z=torch.tensor([-3.,-.2,.7,2.],requires_grad=True);y=targets([-4.,-1.,2.,5.],1)['targets']
    full=terms(z,y,[.25]*4).sum();g=torch.autograd.grad(full,z)[0]
    losses=[];gradients=[]
    for take in itertools.combinations(range(4),2):
        t=list(take);loss=terms(z[t],[y[i] for i in t],[.5]*2).sum()
        losses.append(loss.detach());gradients.append(torch.autograd.grad(loss,z)[0])
    torch.testing.assert_close(torch.stack(losses).mean(),full)
    torch.testing.assert_close(torch.stack(gradients).mean(0),g)


def test_streaming_actual_parameter_gradients_match_dense_including_tail():
    torch.manual_seed(4);model=torch.nn.Linear(2,1);stream=copy.deepcopy(model);x=torch.randn(7,2)
    cfg=dict(fraction=1.,normal_cap=64,anomaly_cap=64)
    sampled=plan('a',1,7,targets(np.linspace(-4,4,7),1),0,cfg)
    expected=terms(model(x).flatten(),sampled['targets'],sampled['coefficients']).sum()
    (expected*.4).backward();value,info=backward(stream,x,sampled,batch=4,scale=.4)
    assert value==pytest.approx(float(expected.detach())) and info['physical_windows']==8
    for a,b in zip(model.parameters(),stream.parameters()):torch.testing.assert_close(a.grad,b.grad)
    assert sampled==plan('a',1,7,targets(np.linspace(-4,4,7),1),0,cfg)


def test_budget_counts_uniform_caps_and_all_validation_windows():
    rows=[dict(name='n',label=0,split='train'),dict(name='a',label=1,split='train'),dict(name='v',label=1,split='val')]
    meta={name:dict(window_count=n) for name,n in [('n',300),('a',33),('v',17)]}
    b=window_budget(rows,meta,dict(fraction=.25,normal_cap=64,anomaly_cap=64))
    assert b['by_class']=={0:64,1:12} and b['validation_physical_windows']==20
