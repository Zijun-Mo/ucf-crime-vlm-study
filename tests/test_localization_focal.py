import numpy as np
import torch
from openjev_dss.localization_focal import terms,negative_focal,plan,calibration,eligibility


def test_window_loss_and_score_gradient_are_invariant_to_companions():
    def value(values,positive,negative,index):
        z=torch.tensor(values,requires_grad=True)
        losses=terms(z,positive,negative,positive_weight=3.)
        gradient=torch.autograd.grad(losses.sum(),z)[0]
        return losses[index].item(),gradient[index].item()
    expected=value([.7],[.2],[.3],0)
    assert value([-20.,.7,20.],[10.,.2,0.],[0.,.3,99.],1)==expected


def test_horvitz_thompson_matches_full_objective_in_expectation(monkeypatch):
    chosen={'teacher':[0.]*6,'segments':[[0,1,2]],'weights':[1.,.5,.2,0,0,0],
            'background':[0,0,0,1/3,1/3,1/3]}
    cfg=dict(fraction=.25,normal_cap=2,positive_cap=1,background_cap=1)
    a=np.zeros(6);b=np.zeros(6)
    # Enumerate the exact uniform joint sampling distribution, not a flaky
    # Monte-Carlo approximation of the Horvitz--Thompson expectation.
    for positive in range(3):
        for background in range(3,6):
            monkeypatch.setattr('openjev_dss.localization_focal.subset',
                lambda indices,count,name,epoch,seed,stream:[positive if stream=='positive' else background])
            p=plan('fixture',1,6,chosen,0,cfg)
            a[p['indices']]+=p['positive'];b[p['indices']]+=p['negative']
    np.testing.assert_allclose(a/9,[1/3,.5/3,.2/3,0,0,0],atol=1e-12)
    np.testing.assert_allclose(b/9,[0,0,0,1/6,1/6,1/6],atol=1e-12)


def test_focal_gradient_calibration_and_background_weight():
    z=torch.tensor([-10.,0.,5.],requires_grad=True)
    derivative=torch.autograd.grad(negative_focal(z).sum(),z)[0]
    p=torch.sigmoid(z.detach())
    expected=p**2*(2*(1-p)*torch.nn.functional.softplus(z.detach())+p)
    torch.testing.assert_close(derivative,expected)
    selected={'teacher':[3.,-3.],'segments':[[0]],'weights':[1.,0.],'background':[0.,1.]}
    r=calibration([{'name':'n','label':0},{'name':'a','label':1}],{'n':[0.,1.],'a':[3.,-3.]},{'a':selected})
    assert .25<=r['positive_weight']<=8 and r['fixed_before_sampling']


def test_selection_rejects_global_auc_gain_with_local_or_ap_regression():
    base=dict(within_macro_auroc=.8,dev_pooled_ap=.4,video_max_ap=.85)
    assert eligibility(base,base)
    assert not eligibility({**base,'within_macro_auroc':.799},base)
    assert not eligibility({**base,'dev_pooled_ap':.39},base)
