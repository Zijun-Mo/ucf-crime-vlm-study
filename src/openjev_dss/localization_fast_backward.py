"""Exact sparse algebra for frozen selection: one-pass BCE, anchor-only ranking."""
import torch
import torch.nn.functional as F
from .localization_backward import padded_forward
from .localization_objective import objective


def backward_fixed_selection(forward,view,label,selected=None,batch=4,scale=1.,
                             positive_weight=1.,rank_weight=1.,guard=lambda:None):
    n=len(view)
    if n<1:raise ValueError('Empty video')
    if not label:
        total=0.
        for begin in range(0,n,batch):
            guard()
            z=padded_forward(forward,view[begin:begin+batch],batch)
            part=F.softplus(z).sum()/n
            (part*scale).backward();total+=float(part.detach())
        return total,dict(normal=total,positive=0.,rank=0.,windows=n,
            active_windows=n,score_calls=0,backward_calls=(n+batch-1)//batch,
            evaluated_windows=n,loss_equivalent=True,execution='one_pass_normal_bce')
    if selected is None or len(selected['teacher'])!=n:
        raise ValueError('Missing aligned frozen teacher selection')
    # Coefficients exactly reproduce the original mean-within-segment then
    # mean-across-segments BCE, including confidence/coverage attenuation.
    alpha=torch.zeros(n,dtype=torch.float32)
    weights=torch.tensor(selected['weights'],dtype=torch.float32)
    for segment in selected['segments']:
        alpha[segment]=weights[segment]/(len(segment)*len(selected['segments']))
    scores=torch.zeros(n,dtype=torch.float32);rank_upstream=torch.zeros_like(scores)
    anchor_batches=sorted({int(i)//batch for pair in selected['pairs'] for i in pair}) if rank_weight else []
    checked={}
    for group in anchor_batches:
        guard();begin=group*batch
        with torch.enable_grad():
            z=padded_forward(forward,view[begin:begin+batch],batch).detach().cpu()
        scores[begin:begin+len(z)]=z;checked[begin]=z
    rank_value=0.;rank_raw=0.
    if anchor_batches:
        leaf=scores.clone().requires_grad_(True)
        rank,info=objective(leaf,1,selected,positive_weight=0.,rank_weight=rank_weight)
        rank_upstream=torch.autograd.grad(rank*scale,leaf)[0]
        rank_value=float(rank.detach());rank_raw=info['rank']
    active=(alpha!=0)&(positive_weight!=0)
    active|=rank_upstream!=0
    groups=sorted(set((torch.nonzero(active).flatten()//batch).tolist()))
    positive=0.;maximum=0.
    for group in groups:
        guard();begin=group*batch
        with torch.enable_grad():
            z=padded_forward(forward,view[begin:begin+batch],batch)
            if begin in checked:
                expected=checked[begin].to(z.device)
                maximum=max(maximum,float((z.detach()-expected).abs().max()))
                torch.testing.assert_close(z.detach(),expected,rtol=2e-3,atol=2e-3)
            a=alpha[begin:begin+len(z)].to(z.device)
            part=(a*F.softplus(-z)).sum()
            g=rank_upstream[begin:begin+len(z)].to(z.device)
            # Rank's linear surrogate contributes its exact current gradient;
            # report the true hinge value, not this surrogate's scalar value.
            (part*positive_weight*scale+(z*g).sum()).backward()
            positive+=float(part.detach())
    return positive_weight*positive+rank_value,dict(normal=0.,positive=positive,rank=rank_raw,
        windows=n,active_windows=int(active.sum()),score_calls=len(anchor_batches),
        backward_calls=len(groups),evaluated_windows=sum(min(batch,n-g*batch) for g in set(anchor_batches)|set(groups)),
        max_recomputed_score_abs=maximum,loss_equivalent=True,execution='fixed_selection_sparse_gradient')
