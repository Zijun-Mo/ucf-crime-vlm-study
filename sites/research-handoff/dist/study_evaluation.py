"""Dual-level video evaluation with exact, video-clustered pooled bootstrap.

Score-tie histograms make pooled AUROC/AP resampling independent of frame count.
Every bootstrap replicate weights whole videos, including all frames and ties.
"""
import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score


def frame_labels(record):
    y = np.zeros(record['n_frames'], dtype=np.uint8)
    for a, b in record['intervals']:
        if a >= 1 and b >= a:
            y[a - 1:min(len(y), b)] = 1
    return y


def frame_scores(record, starts, logits, duration=4, stride=1):
    if len(starts) != len(logits) or not len(starts):
        raise ValueError('Missing predictions')
    if stride <= 0 or duration <= 0 or not np.isfinite(logits).all():
        raise ValueError('Invalid prediction configuration')
    n, fps = record['n_frames'], record['fps']
    sums, counts = np.zeros(n), np.zeros(n, dtype=np.int32)
    for t, z in zip(starts, logits):
        if t % stride:
            continue
        lo, hi = round(t * fps), min(n, round((t + duration) * fps))
        sums[lo:hi] += z
        counts[lo:hi] += 1
    if not np.all(counts > 0):
        raise ValueError('Incomplete frame coverage')
    return sums / counts


def weighted_auc_ap(positive, negative):
    """Counts at ascending UNIQUE scores; rows are bootstrap replicates."""
    p, n = np.atleast_2d(positive), np.atleast_2d(negative)
    ps, ns = p.sum(axis=1), n.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = (p * (np.cumsum(n, axis=1) - n / 2)).sum(axis=1) / (ps * ns)
        pt = np.cumsum(p[:, ::-1], axis=1)
        total = np.cumsum((p + n)[:, ::-1], axis=1)
        precision = np.divide(pt, total, out=np.zeros_like(pt, dtype=float), where=total > 0)
        ap = (precision * p[:, ::-1]).sum(axis=1) / ps
    return auc, ap


def score_histograms(labels, scores):
    if len(labels) != len(scores) or not len(labels):
        raise ValueError('Invalid video collection')
    lengths = [len(y) for y in labels]
    if any(len(y) != len(z) for y, z in zip(labels, scores)):
        raise ValueError('Label/prediction length mismatch')
    y, z = np.concatenate(labels), np.concatenate(scores)
    unique, rank = np.unique(z, return_inverse=True)
    videos = np.repeat(np.arange(len(labels)), lengths)
    shape = (len(labels), len(unique))
    positive = csr_matrix((np.ones(int(y.sum())), (videos[y == 1], rank[y == 1])), shape=shape)
    negative = csr_matrix((np.ones(int((y == 0).sum())), (videos[y == 0], rank[y == 0])), shape=shape)
    return positive, negative


def histogram_bootstrap(labels, scores, weights, batch=32):
    positive, negative = score_histograms(labels, scores)
    point = weighted_auc_ap(np.asarray(positive.sum(axis=0)), np.asarray(negative.sum(axis=0)))
    rows = []
    for begin in range(0, len(weights), batch):
        w = weights[begin:begin + batch]
        rows.append(np.column_stack(weighted_auc_ap(w @ positive, w @ negative)))
    return np.array([point[0][0], point[1][0]]), np.concatenate(rows)


def resampling_weights(n, repeats=1000, seed=42):
    indices = np.random.default_rng(seed).integers(0, n, size=(repeats, n))
    return np.asarray([np.bincount(row, minlength=n) for row in indices])


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'low': float(np.quantile(finite, .025)), 'high': float(np.quantile(finite, .975)),
            'valid_replicates': len(finite)} if len(finite) else None


def evaluate(records, predictions, key, weights, duration=4, stride=1):
    expected = {r['name'] for r in records}
    if expected != set(predictions):
        raise ValueError('Video identities differ from frozen manifest')
    truth, scores, video_rows = [], [], []
    for r in records:
        p = predictions[r['name']]
        expected_starts = list(range(math.ceil(r['n_frames'] / r['fps'])))
        if p['start_s'] != expected_starts:
            raise ValueError('Window starts differ from frozen one-second timeline')
        y = frame_labels(r)
        z = frame_scores(r, p['start_s'], p[key], duration, stride)
        selected = np.asarray([s for t, s in zip(p['start_s'], p[key]) if t % stride == 0])
        row = dict(name=r['name'], category=r['category'], label=int(r['category'] != 'Normal'),
                   windows=len(selected), frames=len(y), positive_frames=int(y.sum()),
                   max=float(selected.max()), mean=float(selected.mean()),
                   topk=float(np.sort(selected)[-max(1, math.ceil(len(selected) / 8)):].mean()),
                   within_auroc=None, within_ap=None)
        if 0 < y.sum() < len(y):
            row.update(within_auroc=float(roc_auc_score(y, z)), within_ap=float(average_precision_score(y, z)))
        truth.append(y)
        scores.append(z)
        video_rows.append(row)
    point, boot = histogram_bootstrap(truth, scores, weights)
    values = {'pooled_auroc': point[0], 'pooled_ap': point[1]}
    samples = {'pooled_auroc': boot[:, 0], 'pooled_ap': boot[:, 1]}
    for aggregate in ['max', 'mean', 'topk']:
        yv = [np.array([r['label']]) for r in video_rows]
        zv = [np.array([r[aggregate]]) for r in video_rows]
        point, boot = histogram_bootstrap(yv, zv, weights)
        for j, metric in enumerate(['auroc', 'ap']):
            name = f'video_{aggregate}_{metric}'
            values[name], samples[name] = point[j], boot[:, j]
    eligible = np.array([r['within_auroc'] is not None for r in video_rows])
    for metric in ['auroc', 'ap']:
        per_video = np.array([r['within_' + metric] for r in video_rows if r['within_' + metric] is not None])
        w = weights[:, eligible]
        name = 'within_macro_' + metric
        values[name] = float(per_video.mean())
        samples[name] = (w @ per_video) / w.sum(axis=1)
    report = {'videos': len(records), 'frames': sum(map(len, truth)),
              'eligible_within_videos': [r['name'] for r in video_rows if r['within_auroc'] is not None],
              'duration_s': duration, 'stride_s': stride,
              'metrics': {name: {'value': float(value), 'ci95': interval(samples[name])} for name, value in values.items()},
              'per_video': video_rows}
    return report, samples
