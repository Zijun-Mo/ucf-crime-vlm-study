"""Rebuild all 12 historical pooled metrics from saved window predictions.

CPU only. No downloads, training, calibration, or repeated bootstrap fitting.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def rebuild(records, folder, key, stride, fingerprint=None, selection=None):
    assert {p.stem for p in folder.glob('*.json')} == {r['name'] for r in records}
    labels, predictions = [], []
    windows = 0
    for record in records:
        pred = read(folder / (record['name'] + '.json'))
        n = record['n_frames']
        assert record['fps'] == 30
        starts = list(range(math.ceil(n / 30)))
        assert pred['start_s'] == starts and len(pred[key]) == len(starts)
        assert all(math.isfinite(z) for z in pred[key])
        if fingerprint:
            assert pred['fingerprint'] == fingerprint
            assert pred['source_sha256'] == record['sha256']
        if selection:
            assert pred['selection_sha256'] == selection['sha256']
        y = np.zeros(n, dtype=np.uint8)
        for lo, hi in record['intervals']:
            if lo >= 0 and hi >= lo:
                y[max(0, lo - 1):min(n, hi)] = 1
        # Independent active-window implementation, then expand to original frames.
        seconds = []
        for t in starts:
            active = [i for i in range(max(0, t - 3), t + 1) if i % stride == 0]
            assert active
            seconds.append(math.fsum(pred[key][i] for i in active) / len(active))
        labels.append(y)
        predictions.append(np.repeat(np.asarray(seconds), 30)[:n])
        windows += sum(i % stride == 0 for i in starts)
    y, z = np.concatenate(labels), np.concatenate(predictions)
    assert len(records) == 290 and len(y) == 1111808 and int(y.sum()) == 84338
    return y, z, windows

def verify():
    formal = ROOT / 'results/formal_stride1'
    tuned = ROOT / 'results/finetuned_test'
    records = read(formal / 'manifest.json')['videos']
    config = read(formal / 'config.json')
    fingerprint = config.pop('fingerprint')
    assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == fingerprint
    checks = []
    for row in read(formal / 'metrics.json')['metrics']:
        model, stride = row['model'], row['stride_s']
        y, z, windows = rebuild(records, formal / model, 'scores', stride, fingerprint)
        with np.load(formal / f'{model}_stride{stride}_frame_scores.npz', allow_pickle=False) as exported:
            assert np.array_equal(y, exported['labels'])
            assert np.allclose(z, exported['scores'], atol=1e-9, rtol=1e-9)
        assert windows == row['windows']
        auc, ap = float(roc_auc_score(y, z)), float(average_precision_score(y, z))
        assert abs(auc - row['auroc']) < 1e-10
        assert abs(ap - row['average_precision']) < 1e-10
        checks.append(dict(model=model, stride_s=stride, windows=windows, auroc=auc, ap=ap, passed=True))
    report = read(tuned / 'summary.json')
    lock = read(tuned / 'selection_lock.json')
    assert report['complete'] and not report['calibration_performed']
    assert report['checkpoint_sha256'] == lock['checkpoints']
    for stride in (1, 2, 4):
        for arm in ('A', 'B'):
            y, z, windows = rebuild(records, tuned / arm, 'raw_logits', stride, selection=lock)
            reference = report['paired_report']['strides'][str(stride)][arm]['all_frames']
            auc, ap = float(roc_auc_score(y, z)), float(average_precision_score(y, z))
            assert abs(auc - reference['auroc']) < 1e-10
            assert abs(ap - reference['ap']) < 1e-10
            checks.append(dict(model='finetuned_' + arm, stride_s=stride, windows=windows, auroc=auc, ap=ap, passed=True))
    return {'videos_per_model': 290, 'frames_per_model': 1111808, 'checks': checks,
            'scope': 'CPU metric reconstruction only; original GPU inference and bootstrap CIs were not rerun'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['checks']:
        print(f"PASS {row['model']:18s} stride={row['stride_s']} AUROC={row['auroc']:.6%} AP={row['ap']:.6%}")
