import torch
from openjev_dss.controlled_runtime import CandidateReadout, load_trainable_snapshot


def test_candidate_readout_matches_full_logits_and_gradients():
    torch.manual_seed(42)
    for bias in [True, False]:
        head = torch.nn.Linear(7, 13, bias=bias)
        head.requires_grad_(False)
        x = torch.randn(4, 7, requires_grad=True)
        full = head(x)
        expected = full[:, 3] - full[:, 8]
        expected.sum().backward()
        gradient = x.grad.clone()
        x.grad = None
        readout = CandidateReadout(head, 3, 8)
        actual = readout(x).flatten()
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, gradient)
        assert not list(readout.parameters())


def test_snapshot_restores_finetuned_values_when_inference_parameters_are_frozen():
    model = torch.nn.Linear(3, 2)
    head = torch.nn.Linear(2, 1)
    model.requires_grad_(False)
    head.requires_grad_(False)
    state = {'model.' + n: torch.full_like(p, 1.25) for n, p in model.named_parameters()}
    state.update({'head.' + n: torch.full_like(p, -.75) for n, p in head.named_parameters()})
    load_trainable_snapshot(model, head, state)
    for p in model.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 1.25))
        assert not p.requires_grad
    for p in head.parameters():
        torch.testing.assert_close(p, torch.full_like(p, -.75))
        assert not p.requires_grad
