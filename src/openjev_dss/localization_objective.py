"""Frozen-teacher selection and within-video ranking; no temporal training labels."""
import math
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SELECTION = dict(normal_quantile=.99, bic_gain=10., min_cluster_gap=.5,
                         soft_coverage_fraction=.25, rank_anchors=32, rank_margin=.5)


def normal_reference(scores, quantile=.99):
    """Equal video mass: long normal videos must not dominate calibration."""
    if not scores or not .5 < quantile < 1:
        raise ValueError('Missing normal training scores or invalid quantile')
    values = [np.asarray(z, dtype=np.float64).ravel() for z in scores]
    if any(not len(z) or not np.isfinite(z).all() for z in values):
        raise ValueError('Invalid normal scores')
    z = np.concatenate(values)
    w = np.concatenate([np.full(len(x), 1 / (len(values) * len(x))) for x in values])
    order = np.argsort(z, kind='stable'); z, w = z[order], w[order]
    cdf = np.cumsum(w)
    def q(p): return float(z[min(len(z)-1, np.searchsorted(cdf, p))])
    return dict(threshold=q(quantile), median=q(.5), scale=max((q(.75)-q(.25))/1.349, .1),
                quantile=quantile, videos=len(values), windows=len(z), source='normal_train_only')


def split_scores(values, scale, config):
    """Optimal two contiguous score clusters, accepted only with BIC separation.

    This is a score partition, not a claim that clusters are true anomalies.
    BIC uses one common Gaussian variance and a fixed extra-parameter penalty.
    """
    t = np.asarray(values, dtype=np.float64)
    if len(t) < 4 or not np.isfinite(t).all():
        return None
    order = np.argsort(t, kind='stable'); x = t[order]
    if x[-1] - x[0] < config['min_cluster_gap'] * scale:
        return None
    n = len(x); c = np.cumsum(x); c2 = np.cumsum(x*x)
    k = np.arange(1, n)
    sse = np.maximum(c2[:-1]-c[:-1]**2/k, 0)
    sse += np.maximum(c2[-1]-c2[:-1]-(c[-1]-c[:-1])**2/(n-k), 0)
    sse[x[:-1] == x[1:]] = np.inf
    cut = int(np.argmin(sse))+1
    total = max(float(c2[-1]-c[-1]**2/n), 1e-12)
    residual = max(float(sse[cut-1]), 1e-12)
    gain = n * math.log(total/residual) - 2*math.log(n)
    gap = float(x[cut:].mean()-x[:cut].mean())
    if gain < config['bic_gain'] or gap < config['min_cluster_gap'] * scale:
        return None
    return dict(low=order[:cut].tolist(), high=order[cut:].tolist(), bic_gain=gain, gap=gap)


def _anchors(indices, maximum):
    # Distributed time anchors, not a newly sampled positive MIL bag.
    x = sorted(indices)
    if len(x) <= maximum: return x
    return [x[i] for i in np.linspace(0, len(x)-1, maximum, dtype=int)]


def selection(values, reference, config=None):
    config = {**DEFAULT_SELECTION, **(config or {})}
    t = np.asarray(values, dtype=np.float64).ravel()
    if not len(t) or not np.isfinite(t).all(): raise ValueError('Invalid teacher scores')
    scale = reference['scale']; partition = split_scores(t, scale, config)
    weights = np.zeros(len(t)); segments = []
    if partition:
        candidates = sorted(i for i in partition['high'] if t[i] >= reference['threshold'])
        for i in candidates:
            if not segments or i > segments[-1][-1] + 4: segments.append([])
            segments[-1].append(i)
        if candidates:
            # Connected 4s windows form a union, including gaps < 4 seconds.
            # Assign its clipped length equally to the contributing windows.
            costs = np.zeros(len(t))
            for segment in segments:
                costs[segment] = (min(len(t),segment[-1]+4)-segment[0])/len(segment)
            u = (t[candidates]-reference['threshold'])/scale
            budget = config['soft_coverage_fraction'] * len(t)
            def gate(shift): return 1/(1+np.exp(-np.clip(u-shift, -60, 60)))
            lo, hi = 0., 100.
            if float((costs[candidates]*gate(0)).sum()) > budget:
                for _ in range(60):
                    mid=(lo+hi)/2
                    if float((costs[candidates]*gate(mid)).sum()) > budget: lo=mid
                    else: hi=mid
                shift=hi
            else: shift=0.
            weights[candidates] = gate(shift)
            soft_coverage=float((weights*costs).sum())
        else: soft_coverage=0.
        hi = _anchors(partition['high'], config['rank_anchors'])
        lo = _anchors(partition['low'], config['rank_anchors'])
        pairs = [(i,j) for i in hi for j in lo if t[i]-t[j] >= scale]
    else: pairs=[];soft_coverage=0.
    return dict(weights=weights.tolist(), segments=segments, pairs=pairs,
                teacher=t.tolist(), scale=scale, rank_margin=config['rank_margin'],
                summary=dict(windows=len(t), positive_windows=int((weights>0).sum()),
                             positive_segments=len(segments), soft_count=float(weights.sum()),
                             soft_coverage_s=soft_coverage, rank_pairs=len(pairs),
                             accepted_partition=partition is not None,
                             bic_gain=partition['bic_gain'] if partition else None))


def objective(logits, label, selected=None, positive_weight=1., rank_weight=1.):
    z=logits.float().flatten()
    if not len(z) or not torch.isfinite(z).all(): raise ValueError('Invalid student scores')
    zero=z.sum()*0
    if not label:
        normal=F.softplus(z).mean()
        return normal, dict(normal=float(normal.detach()), positive=0., rank=0.)
    if selected is None or len(selected['teacher']) != len(z):
        raise ValueError('Missing or misaligned frozen selection')
    weights=z.new_tensor(selected['weights']); teacher=z.new_tensor(selected['teacher'])
    # Mean over windows within each segment, then over segments. Do not divide
    # by sum(weights): that would cancel the soft coverage budget's attenuation.
    pieces=[(weights[segment]*F.softplus(-z[segment])).mean() for segment in selected['segments']]
    positive=torch.stack(pieces).mean() if pieces else zero
    if selected['pairs']:
        pairs=torch.as_tensor(selected['pairs'],device=z.device,dtype=torch.long)
        i,j=pairs[:,0],pairs[:,1];scale=selected['scale']
        margin=selected['rank_margin']*(teacher[i]-teacher[j])/scale
        rank=F.relu(margin-(z[i]-z[j])/scale).mean()
    else:rank=zero
    total=positive_weight*positive+rank_weight*rank
    return total,dict(normal=0.,positive=float(positive.detach()),rank=float(rank.detach()))
