"""Bounded streaming score-leaf differentiation over ALL video windows."""
import torch


def padded_forward(forward, features, batch=4):
    n=len(features)
    if n < batch:
        features=torch.cat([features,features[:1].expand(batch-n,*features.shape[1:])])
    return forward(features).float().flatten()[:n]


def stream_scores(forward, view, batch=4, guard=lambda:None, grad_mode=False):
    parts=[]
    for i in range(0,len(view),batch):
        guard()
        # Preserve the physical path used for training's recomputation.
        with torch.set_grad_enabled(grad_mode):
            z=padded_forward(forward,view[i:i+batch],batch)
            parts.append(z.detach().cpu())
    return torch.cat(parts)


def backward_video(forward, view, loss_fn, batch=4, scale=1., guard=lambda:None):
    scores=stream_scores(forward,view,batch,guard,grad_mode=True)
    leaf=scores.clone().requires_grad_(True)
    loss,info=loss_fn(leaf)
    if not torch.isfinite(loss):raise ValueError('Nonfinite localization objective')
    upstream=torch.autograd.grad(loss*scale,leaf)[0]
    active=0; calls=0;maximum=0.
    for i in range(0,len(view),batch):
        g=upstream[i:i+batch]
        if not torch.any(g != 0):continue
        guard()
        with torch.enable_grad():
            current=padded_forward(forward,view[i:i+batch],batch)
            expected=scores[i:i+batch].to(current.device)
            maximum=max(maximum,float((current.detach()-expected).abs().max()))
            torch.testing.assert_close(current.detach(),expected,rtol=2e-3,atol=2e-3)
            current.backward(g.to(current.device))
        active+=int((g!=0).sum());calls+=1
    return float(loss.detach()),dict(**info,windows=len(view),active_windows=active,
        score_calls=(len(view)+batch-1)//batch,backward_calls=calls,
        max_recomputed_score_abs=maximum,all_windows_scored=True)
