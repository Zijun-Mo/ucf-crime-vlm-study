"""Exact first-order score-leaf chain rule, with deterministic forwards."""
import torch
from .losses import loss

def two_pass(forward, windows, labels, dss, scale=1., chunk=1, score_batch=None,
             pad_batches=False):
    # Forwards must be independent across windows, dropout=0, cache=False.
    # Never create tensors in inference_mode: they may be saved for backward.
    score_batch=chunk if score_batch is None else score_batch
    if chunk<1 or score_batch<1:raise ValueError('Batch sizes must be positive')
    sizes=[len(w) for w in windows]
    if not sizes or min(sizes)<1:raise ValueError('Empty bag')
    # Pack across bags: batching is a physical execution detail, not a change to
    # video labels, top-k selection, or loss normalization. Padding has zero weight.
    packed=torch.cat(windows)
    def run(w,batch):
        n=len(w)
        if pad_batches and n<batch:
            w=torch.cat([w,w[:1].expand(batch-n,*w.shape[1:])])
        return forward(w).float().flatten()[:n]
    with torch.no_grad():
        all_scores=torch.cat([run(packed[i:i+score_batch],score_batch)
                              for i in range(0,len(packed),score_batch)])
        scores=list(all_scores.split(sizes))
    leaves = [z.detach().clone().requires_grad_(True) for z in scores]
    objective, info = loss(leaves,labels,dss)
    gradients = torch.autograd.grad(objective*scale,leaves)
    g=torch.cat(gradients);idx=torch.nonzero(g!=0,as_tuple=True)[0];active=len(idx)
    for i in range(0,active,chunk):
        take=idx[i:i+chunk]
        current=run(packed[take.to(packed.device)],chunk)
        torch.testing.assert_close(current.detach(),all_scores[take],rtol=2e-3,atol=2e-3)
        current.backward(g[take])
    return float(objective.detach()), {'mil':float(info['mil'].detach()),
        'ss':float(info['ss'].detach()),'bags':info['bags'], 'active_windows':active,
        'score_batch':score_batch,'backward_batch':chunk,
        'score_calls':(len(packed)+score_batch-1)//score_batch,
        'backward_calls':(active+chunk-1)//chunk}

def trainable_state(model,head):
    return {**{'model.'+k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad},
            **{'head.'+k:p.detach().cpu().clone() for k,p in head.named_parameters()}}

def restore(model,head,state):
    with torch.no_grad():
        for k,p in model.named_parameters():
            if p.requires_grad:p.copy_(state['model.'+k].to(p.device))
        for k,p in head.named_parameters():p.copy_(state['head.'+k].to(p.device))
