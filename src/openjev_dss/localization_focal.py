"""Frozen three-state pseudo labels and independent focal/BCE window losses.

All selection, mass normalization and positive calibration occur before student
sampling. No batch ranks, sampled-bag targets, or student-score normalizers exist.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .localization_objective import selection,split_scores
from .localization_pointwise import coefficients,subset
from .localization_backward import padded_forward


def select(values,reference,config):
    chosen=selection(values,reference,config)
    partition=split_scores(values,reference['scale'],config)
    background=np.zeros(len(values),dtype=np.float64)
    if partition:
        eligible=[i for i in partition['low'] if values[i]<=reference['median']]
        if eligible:background[eligible]=1/len(eligible)
    chosen['background']=background.tolist()
    chosen['summary']['background_windows']=int((background>0).sum())
    chosen['summary']['uncertain_windows']=int(((coefficients(chosen)==0)&(background==0)).sum())
    assert not np.any((coefficients(chosen)>0)&(background>0))
    return chosen


def negative_focal(logits,gamma=2.):
    z=logits.float()
    return torch.sigmoid(z).pow(gamma)*F.softplus(z)


def terms(logits,positive,negative,positive_weight=1.,gamma=2.):
    z=logits.float().flatten()
    a=torch.as_tensor(positive,device=z.device,dtype=z.dtype)
    b=torch.as_tensor(negative,device=z.device,dtype=z.dtype)
    if a.shape!=z.shape or b.shape!=z.shape:raise ValueError('Coefficient shape mismatch')
    return positive_weight*a*F.softplus(-z)+b*negative_focal(z,gamma)


def plan(name,label,n,selected,epoch,config,seed=42,background_weight=.5):
    a=np.zeros(n);b=np.zeros(n)
    if not label:
        count=min(config['normal_cap'],max(1,math.ceil(n*config['fraction'])))
        take=subset(range(n),count,name,epoch,seed,'normal')
        b[take]=1/count
    else:
        positive=coefficients(selected);background=np.asarray(selected['background'])
        for full,target,cap,stream in [(positive,a,config['positive_cap'],'positive'),
                                      (background*background_weight,b,config['background_cap'],'background')]:
            eligible=np.flatnonzero(full>0);count=min(cap,len(eligible))
            if count:
                take=subset(eligible,count,name,epoch,seed,stream)
                target[take]=full[take]*len(eligible)/count
    indices=np.flatnonzero((a>0)|(b>0)).tolist()
    return dict(indices=indices,positive=a[indices].tolist(),negative=b[indices].tolist(),
        label=int(label),total_windows=n,positive_draws=int((a>0).sum()),
        background_draws=int((b>0).sum()) if label else 0,window_independent=True)


def backward(forward,view,sampled,batch=4,scale=1.,positive_weight=1.,gamma=2.,guard=lambda:None):
    indices=sampled['indices'];total=0.
    for begin in range(0,len(indices),batch):
        guard();take=indices[begin:begin+batch]
        z=padded_forward(forward,view[take],batch)
        loss=terms(z,sampled['positive'][begin:begin+batch],sampled['negative'][begin:begin+batch],positive_weight,gamma).sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite focal localization loss')
        (loss*scale).backward();total+=float(loss.detach())
    return total,dict(windows=sampled['total_windows'],sampled_windows=len(indices),
        physical_windows=batch*math.ceil(len(indices)/batch),positive_draws=sampled['positive_draws'],
        background_draws=sampled['background_draws'],window_independent=True,
        execution='fixed_three_state_pointwise_one_pass')


def calibration(records,teachers,selections,gamma=2.,background_weight=.5):
    """Class/video-balanced initial dL/dz masses; no LM backward or updates."""
    counts={y:sum(r['label']==y for r in records) for y in [0,1]}
    positive=negative=0.
    for row in records:
        name=row['name'];z=torch.tensor(teachers[name],dtype=torch.float64)
        p=torch.sigmoid(z);softplus=F.softplus(z)
        dn=p.pow(gamma)*(gamma*(1-p)*softplus+p)
        weight=1/(2*counts[row['label']])
        if row['label']:
            selected=selections[name]
            positive+=weight*float((torch.as_tensor(coefficients(selected))*(1-p)).sum())
            negative+=weight*background_weight*float((torch.as_tensor(selected['background'])*dn).sum())
        else:negative+=weight*float(dn.mean())
    if positive<=0:raise ValueError('No positive pseudo-label gradient mass; training forbidden')
    ratio=negative/positive
    return dict(positive_weight=float(np.clip(ratio,.25,8)),unclipped_ratio=ratio,
        initial_positive_score_gradient_mass=positive,initial_negative_score_gradient_mass=negative,
        source='complete frozen training teacher cache only',fixed_before_sampling=True,
        test_used=False,validation_used=False)


def eligibility(result,baseline):
    return all(result[k]>=baseline[k] for k in ['within_macro_auroc','dev_pooled_ap','video_max_ap'])
