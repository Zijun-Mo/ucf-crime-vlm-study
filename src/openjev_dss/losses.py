"""FP32, differentiable score-sensitive MIL; DSS adaptation, not reproduction."""
import math
import torch
import torch.nn.functional as F

def indices(p, k, descending):
    # Stable ties and discrete selection; score weights retain gradients.
    return torch.argsort(p.detach(), descending=descending, stable=True)[:k]

def bag_loss(logits, label, dss=False):
    z = logits.float().flatten()
    if not len(z): raise ValueError('Empty bag')
    k = max(1, math.ceil(len(z)/8))
    lp, ln = F.logsigmoid(z), F.logsigmoid(-z)
    p = lp.exp()
    top = indices(p,k,True)
    lq = torch.logsumexp(lp[top],0)-math.log(k)
    lnotq = torch.logsumexp(ln[top],0)-math.log(k)
    mil = -lq if label else -lnotq
    ss = z.sum()*0
    suppressed = torch.zeros_like(p,dtype=torch.bool)
    changed = False
    it = ib = None
    if dss and label:
        suppressed = p.detach() > .85
        maximum = indices(p,1,True)[0]
        spl = lp[maximum] + math.log(.55)
        # Suppressed scores <= .55: complement evaluation is stable.
        sln = torch.log1p(-spl.exp())
        lp2 = torch.where(suppressed,spl,lp)
        ln2 = torch.where(suppressed,sln,ln)
        p2 = lp2.exp()
        it, ib = indices(p2,k,True), indices(p2,k,False)
        la = torch.logsumexp(lp2[it],0)-math.log(k)
        lb = torch.logsumexp(lp2[ib],0)-math.log(k)
        lnotb = torch.logsumexp(ln2[ib],0)-math.log(k)
        # Stable 1-a; BOTH score weights participate in differentiation.
        oneminusa = (torch.logsumexp(ln2[it],0)-math.log(k)).exp()
        ss = -oneminusa*la - lb.exp()*lnotb
        changed = not torch.equal(top.sort().values,it.sort().values)
    return mil, ss, {'n':len(z),'k':k,'suppressed':int(suppressed.sum()),'top_changed':changed,'mil_top':top.cpu().tolist(),
        'ss_top':it.cpu().tolist() if it is not None else [],
        'ss_bottom':ib.cpu().tolist() if ib is not None else []}

def loss(bags, labels, dss=False):
    if len(bags)!=len(labels) or not bags: raise ValueError('Invalid bags')
    pieces = [bag_loss(z,int(y),dss) for z,y in zip(bags,labels)]
    mil = torch.stack([x[0] for x in pieces]).mean()
    positive = [x[1] for x,y in zip(pieces,labels) if y]
    ss = torch.stack(positive).mean() if positive else mil*0
    return mil+ss, {'mil':mil,'ss':ss,'bags':[x[2] for x in pieces]}
