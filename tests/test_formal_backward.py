import copy
import math
import pytest
import torch
from openjev_dss.formal_backward import two_pass
from openjev_dss.losses import loss


@pytest.mark.parametrize('grad_mode_offset',[False,True])
def test_matches_dense_microbatch_autograd_with_original_slots_and_tail_padding(grad_mode_offset):
    model=torch.nn.Linear(1,1)
    with torch.no_grad():model.weight.fill_(.15);model.bias.fill_(-.8)
    candidate=copy.deepcopy(model)
    bags=[torch.arange(1,8,dtype=torch.float32)[:,None],torch.arange(8,19,dtype=torch.float32)[:,None]]
    packed=torch.cat(bags);original=[];z=[]
    for i in range(0,len(packed),4):
        batch=packed[i:i+4];n=len(batch)
        if n<4:batch=torch.cat([batch,batch[:1].expand(4-n,1)])
        original.append(batch.clone());z.append(model(batch).flatten()[:n])
    dense_objective,_=loss(list(torch.cat(z).split([7,11])),[0,1],False)
    dense_objective.backward()
    expected=torch.cat([p.grad.flatten() for p in model.parameters()])
    calls=[]
    def forward(batch):
        calls.append((batch.clone(),torch.is_grad_enabled()))
        # A forward implementation can take a different numerical path under
        # no_grad; the dense-reference path must be used for score derivatives.
        return candidate(batch)+(0.05 if grad_mode_offset and not torch.is_grad_enabled() else 0.)
    objective,info=two_pass(forward,bags,[0,1])
    actual=torch.cat([p.grad.flatten() for p in candidate.parameters()])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    assert objective==pytest.approx(float(dense_objective.detach()))
    assert info['backward_calls']<math.ceil(len(packed)/4)
    assert info['backward_calls']>math.ceil(info['active_windows']/4)
    assert all(enabled for _,enabled in calls)
    assert all(any(torch.equal(batch,reference) for reference in original) for batch,_ in calls)
