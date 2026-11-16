"""Score-leaf MIL with the same physical batches and grad mode as dense AD.

Only one physical batch's graph is retained. Zero-gradient windows keep their
original batch positions; whole batches with zero upstream gradient are skipped.
The historical training.two_pass implementation is deliberately untouched.
"""
import torch
from .losses import loss


def two_pass(forward, windows, labels, dss=False, scale=1., chunk=4, score_batch=4,
             pad_batches=True, observer=None):
    assert chunk==score_batch and chunk>0 and pad_batches
    sizes=[len(w) for w in windows]
    if not sizes or min(sizes)<1:raise ValueError('Empty bag')
    packed=torch.cat(windows)

    def run(batch):
        n=len(batch)
        if n<chunk:batch=torch.cat([batch,batch[:1].expand(chunk-n,*batch.shape[1:])])
        return forward(batch).float().flatten()[:n]

    # no_grad can select different numerical forward kernels. Build a transient
    # physical-batch graph and detach immediately; never retain all bag graphs.
    with torch.enable_grad():
        all_scores=torch.cat([run(packed[i:i+chunk]).detach() for i in range(0,len(packed),chunk)])
    leaf=all_scores.detach().clone().requires_grad_(True)
    # Dense AD passes views of ONE packed score vector into the bag loss.
    # Preserve its offsets/layout instead of allocating each bag separately.
    objective,info=loss(list(leaf.split(sizes)),labels,dss)
    g=torch.autograd.grad(objective*scale,leaf)[0]
    if observer is not None:observer(all_scores.detach(),g.detach())
    calls=0;maximum=0.
    for i in range(0,len(packed),chunk):
        upstream=g[i:i+chunk]
        if not torch.any(upstream!=0):continue
        # Repacking selected windows changes BF16 reduction/layout paths. Keep
        # all original slots, including windows with zero upstream derivative.
        with torch.enable_grad():
            current=run(packed[i:i+chunk])
            maximum=max(maximum,float((current.detach()-all_scores[i:i+chunk]).abs().max()))
            torch.testing.assert_close(current.detach(),all_scores[i:i+chunk],rtol=2e-3,atol=2e-3)
            current.backward(upstream)
        calls+=1
    return float(objective.detach()),{'mil':float(info['mil'].detach()),'ss':float(info['ss'].detach()),
        'bags':info['bags'],'active_windows':int((g!=0).sum()),'score_batch':score_batch,'backward_batch':chunk,
        'score_calls':(len(packed)+chunk-1)//chunk,'backward_calls':calls,'backward_window_slots':calls*chunk,
        'original_batch_positions_preserved':True,'score_forward_grad_enabled':True,'max_recomputed_score_abs':maximum}
