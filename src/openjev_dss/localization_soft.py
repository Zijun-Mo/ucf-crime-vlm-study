"""Frozen per-video min-max soft targets; independent, unbiased window BCE.

Borrow the target normalization from MIST, not its complete architecture or MIL
teacher. Targets use sigmoid(native Qwen logits) from the *entire* video before
sampling. No student, minibatch or sampled-bag extrema enter the objective.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .localization_backward import padded_forward
from .localization_pointwise import subset


def targets(values, label, minimum_range=1e-12, *, mode='minmax', name=None, seed=42):
    z=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(z) or not np.isfinite(z).all():raise ValueError('Invalid complete teacher logits')
    p=torch.sigmoid(torch.as_tensor(z)).numpy()
    span=float(np.ptp(p));active=not label or span>minimum_range
    y=(p-p.min())/span if label and active else np.zeros(len(z))
    if mode not in {'minmax', 'permuted_minmax', 'raw_probability'}:
        raise ValueError(f'Unknown soft target mode: {mode}')
    if label and active and mode == 'permuted_minmax':
        if not name:
            raise ValueError('Video identity required for fixed target permutation')
        # Independent stream: this cannot affect the student window sampler.
        import hashlib
        key = hashlib.sha256(f'{seed}:e9-target-permutation:{name}'.encode()).digest()
        y = y[np.random.default_rng(int.from_bytes(key[:8], 'little')).permutation(len(y))]
    elif label and active and mode == 'raw_probability':
        y = p
    return dict(targets=y.tolist(),active=active,teacher_probability_min=float(p.min()),
        teacher_probability_max=float(p.max()),probability_range=span,
        summary=dict(windows=len(z),positive_windows=int((y>0).sum()),
            mean_target=float(y.mean()),abstained=not active),
        provenance=f'frozen full-video sigmoid teacher scores; {mode}; no temporal smoothing')


def terms(logits, labels, coefficients):
    z=logits.float().flatten()
    y=torch.as_tensor(labels,device=z.device,dtype=z.dtype)
    a=torch.as_tensor(coefficients,device=z.device,dtype=z.dtype)
    if z.shape!=y.shape or z.shape!=a.shape:raise ValueError('Soft target shape mismatch')
    if not bool(((y>=0)&(y<=1)).all()):raise ValueError('Targets outside [0,1]')
    return a*F.binary_cross_entropy_with_logits(z,y,reduction='none')


def plan(name,label,n,selected,epoch,config,seed=42):
    if n<=0 or not 0<config['fraction']<=1:raise ValueError('Invalid sampling inputs')
    if label and (len(selected['targets'])!=n):raise ValueError('Incomplete fixed targets')
    active=not label or selected['active']
    cap=config['anomaly_cap'] if label else config['normal_cap']
    count=min(cap,max(1,math.ceil(n*config['fraction']))) if active else 0
    take=subset(range(n),count,name,epoch,seed,'soft_bce') if count else []
    # Full coefficient 1/n divided by fixed inclusion probability count/n.
    return dict(indices=take,targets=[selected['targets'][i] for i in take] if label else [0.]*count,
        coefficients=[1/count]*count if count else [],inclusion_probability=count/n,
        total_windows=n,label=int(label),window_independent=True,active=active)


def backward(forward,view,sampled,batch=4,scale=1.,guard=lambda:None):
    indices=sampled['indices'];total=0.
    for begin in range(0,len(indices),batch):
        guard();z=padded_forward(forward,view[indices[begin:begin+batch]],batch)
        loss=terms(z,sampled['targets'][begin:begin+batch],sampled['coefficients'][begin:begin+batch]).sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite soft localization BCE')
        (loss*scale).backward();total+=float(loss.detach())
    return total,dict(windows=sampled['total_windows'],sampled_windows=len(indices),
        physical_windows=batch*math.ceil(len(indices)/batch),window_independent=True,
        inclusion_probability=sampled['inclusion_probability'],execution='frozen_minmax_soft_bce_one_pass')


def window_budget(records,metadata,config,batch=4):
    by_class={0:0,1:0};validation=sampled=0
    for r in records:
        n=metadata[r['name']]['window_count']
        if r['split']=='val':validation+=batch*math.ceil(n/batch)
        elif r['split']=='train':
            cap=config['anomaly_cap'] if r['label'] else config['normal_cap']
            m=min(cap,max(1,math.ceil(n*config['fraction'])))
            by_class[r['label']]+=batch*math.ceil(m/batch);sampled+=m
    return dict(training_physical_windows_upper=sum(by_class.values()),
        validation_physical_windows=validation,sampled_windows_upper=sampled,by_class=by_class)
