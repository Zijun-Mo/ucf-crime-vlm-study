"""Independent window objectives with frozen full-video teacher coefficients."""
import hashlib
import math
import numpy as np
import torch
import torch.nn.functional as F
from .localization_backward import padded_forward


def coefficients(selected):
    """Fixed segment/video normalization; never normalize a sampled bag."""
    alpha=np.zeros(len(selected['teacher']),dtype=np.float64)
    for segment in selected['segments']:
        alpha[segment]=np.asarray(selected['weights'])[segment]/(len(segment)*len(selected['segments']))
    return alpha


def subset(indices,count,name,epoch,seed,stream):
    """Uniform subset, deterministic on resume, independent of other videos."""
    if not len(indices):return []
    count=min(len(indices),count)
    key=f'{seed}:{name}:{epoch}:{stream}'
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big'))
    return sorted(rng.choice(np.asarray(indices),count,replace=False).tolist())


def plan(name,label,n,selected,epoch,config,seed=42):
    """Horvitz--Thompson coefficients estimate the fixed full-video objective."""
    fraction=config['fraction'];assert 0<fraction<=1 and n>0
    count=min(config['normal_cap'],max(1,math.ceil(n*fraction)))
    bce=np.zeros(n,dtype=np.float64);anchor=np.zeros(n,dtype=np.float64)
    if not label:
        take=subset(range(n),count,name,epoch,seed,'normal')
        bce[take]=1/count
    else:
        alpha=coefficients(selected);eligible=np.flatnonzero(alpha>0)
        count_pos=min(config['positive_cap'],len(eligible))
        if count_pos:
            take=subset(eligible,count_pos,name,epoch,seed,'positive')
            bce[take]=alpha[take]*len(eligible)/count_pos
        count_anchor=min(config['anchor_cap'],max(1,math.ceil(n*fraction)))
        take=subset(range(n),count_anchor,name,epoch,seed,'teacher')
        anchor[take]=1/count_anchor
    indices=np.flatnonzero((bce!=0)|(anchor!=0)).tolist()
    return dict(indices=indices,bce=bce[indices].tolist(),anchor=anchor[indices].tolist(),
        teacher=[selected['teacher'][i] for i in indices] if label else [0.]*len(indices),
        scale=selected['scale'] if label else 1.,total_windows=n,
        positive_draws=int((bce>0).sum()),anchor_draws=int((anchor>0).sum()),
        label=int(label),window_independent=True)


def terms(logits,label,bce,anchor,teacher,teacher_scale,anchor_weight=1.,huber_delta=1.):
    """Each returned term depends only on that window and fixed coefficients."""
    z=logits.float().flatten()
    def tensor(x):return torch.as_tensor(x,device=z.device,dtype=z.dtype)
    a,b,t=tensor(bce),tensor(anchor),tensor(teacher)
    classification=a*F.softplus(-z if label else z)
    residual=(z-t)/teacher_scale
    tether=b*F.huber_loss(residual,torch.zeros_like(residual),delta=huber_delta,reduction='none')
    return classification+anchor_weight*tether


def backward_pointwise(forward,view,sampled,batch=4,scale=1.,anchor_weight=1.,huber_delta=1.,guard=lambda:None):
    indices=sampled['indices'];total=0.
    for begin in range(0,len(indices),batch):
        guard();take=indices[begin:begin+batch]
        z=padded_forward(forward,view[take],batch)
        loss=terms(z,sampled['label'],sampled['bce'][begin:begin+batch],
            sampled['anchor'][begin:begin+batch],sampled['teacher'][begin:begin+batch],
            sampled['scale'],anchor_weight,huber_delta).sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite pointwise loss')
        (loss*scale).backward();total+=float(loss.detach())
    return total,dict(windows=sampled['total_windows'],sampled_windows=len(indices),
        physical_windows=batch*math.ceil(len(indices)/batch),positive_draws=sampled['positive_draws'],
        anchor_draws=sampled['anchor_draws'],score_calls=0,backward_calls=math.ceil(len(indices)/batch),
        window_independent=True,execution='frozen_teacher_pointwise_one_pass')


def window_budget(records,metadata,config,batch=4):
    """A conservative shape-only compute ceiling, before teacher selection."""
    train=val=0;by_class={0:0,1:0};sampled=0
    for r in records:
        if r['split'] not in ['train','val']:continue
        n=metadata[r['name']]['window_count']
        if r['split']=='val':val+=batch*math.ceil(n/batch);continue
        if r['split']!='train':continue
        fraction=max(1,math.ceil(n*config['fraction']))
        m=(min(config['normal_cap'],fraction) if not r['label'] else
           min(n,min(config['positive_cap'],n)+min(config['anchor_cap'],fraction)))
        physical=batch*math.ceil(m/batch);train+=physical;by_class[r['label']]+=physical;sampled+=m
    return dict(training_physical_windows_upper=train,validation_physical_windows=val,
        sampled_windows_upper=sampled,by_class=by_class)
