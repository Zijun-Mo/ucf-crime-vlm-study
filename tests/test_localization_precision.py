"""FP8 projection swaps must preserve native evaluation and frozen state."""
from types import SimpleNamespace
import torch
from openjev_dss.localization_precision import patch_frozen_linears,set_precision


def test_precision_patch_preserves_native_eval_and_trainable_parameters():
    torch.manual_seed(7)
    base=torch.nn.Linear(16,16,bias=False,dtype=torch.bfloat16)
    base.requires_grad_(False)
    adapter=torch.nn.Linear(16,16,bias=False,dtype=torch.bfloat16)
    model=torch.nn.Sequential(base,adapter)
    inputs=torch.randn(5,16,dtype=torch.bfloat16)
    model.eval();expected=model(inputs).detach().clone()
    weights={name:p.detach().clone() for name,p in model.named_parameters()}
    parameters={name:id(p) for name,p in model.named_parameters()}
    modules=patch_frozen_linears(SimpleNamespace(model=model))
    assert [name for name,_ in modules]==['0']
    set_precision(modules,'fp8_full')
    torch.testing.assert_close(model(inputs),expected,rtol=0,atol=0)
    assert {name:id(p) for name,p in model.named_parameters()}==parameters
    for name,p in model.named_parameters():torch.testing.assert_close(p,weights[name],rtol=0,atol=0)
    assert not any('fp8' in name for name in model.state_dict())
    model.train();set_precision(modules,'bf16')
    model(inputs).float().sum().backward()
    assert base.weight.grad is None
    assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad).all()
