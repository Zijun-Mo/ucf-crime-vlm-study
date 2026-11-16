"""Admit historical training features only at the original physical batch size."""
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .runtime import decode_canonical_frames


def training_fixtures(records):
    groups = [[r for r in records if r['split'] == 'train' and r['label'] == y] for y in [0, 1]]
    if any(len(g) < 4 for g in groups):
        raise ValueError('Need four training fixtures per label; never select test videos')
    ordered = [sorted(g, key=lambda r: (r['size'], r['name'])) for g in groups]
    return [g[int(q * len(g))] for q in [0, .25, .5, .75] for g in ordered]


def verify_training_cache(runtime, provider, records, historical_cache, local, guard=lambda: None):
    started = time.time()
    fixtures = training_fixtures(records)
    report = {'status': 'blocked', 'cache_reuse_allowed': False, 'physical_batch': 4,
              'windows_per_fixture': 'original sampled starts, at most 16', 'test_used': False,
              'fixtures': [r['name'] for r in fixtures], 'checks': [],
              'criterion': 'every sampled window must be bitwise identical; original batches up to four, including the original unpadded tail'}
    local = Path(local); local.mkdir(parents=True, exist_ok=True)
    for record in fixtures:
        guard()
        metadata, starts, signature, target = provider.descriptor(record, 0, False)
        if metadata['sha256'] != record['sha256']:
            raise RuntimeError('Training metadata source differs from frozen manifest')
        previous = Path(historical_cache) / target.name
        row = {'name': record['name'], 'cache_file': previous.name, 'source_sha256': record['sha256']}
        report['checks'].append(row)
        cache_started = time.time()
        try:
            saved = provider.cache.get(previous, signature)
        except Exception as error:
            row.update(bitwise_equal=False, reason='historical cache unreadable: '+type(error).__name__)
            break
        row['cache_read_s'] = time.time()-cache_started
        if saved is None:
            row.update(bitwise_equal=False, reason='historical cache missing')
            break
        if not 1 <= len(starts) <= 16 or saved.ndim != 3 or saved.shape[0] != len(starts) or not saved.is_floating_point() or not torch.isfinite(saved).all():
            row.update(bitwise_equal=False, reason='historical training shape/dtype mismatch')
            break
        guard()
        row['sampled_windows'] = len(starts)
        row['cached_dtype'] = str(saved.dtype)
        regenerate_started = time.time()
        canonical = local / record['name']
        try:
            shutil.copyfile(provider.root / 'canonical' / record['name'], canonical)
            if hashlib.sha256(canonical.read_bytes()).hexdigest() != metadata['canonical_sha256']:
                raise RuntimeError('Canonical training video digest mismatch')
            frames = decode_canonical_frames(canonical, metadata['sampled_frames'], starts, False)
            row.update(bitwise_equal=True, max_abs=0., compared_windows=0)
            for begin in range(0, len(starts), 4):
                guard()
                windows = [np.stack([frames[min(2*s+j, metadata['sampled_frames'])] for j in range(8)]) for s in starts[begin:begin+4]]
                value = runtime.vision(windows)
                old = saved[begin:begin+4]
                row['regenerated_dtype'] = str(value.dtype)
                equal = value.dtype == old.dtype and bool(torch.isfinite(value).all()) and torch.equal(value, old)
                difference = float((value.float()-old.float()).abs().max()) if value.shape == old.shape else None
                row.update(bitwise_equal=bool(equal), compared_windows=begin+len(windows))
                if difference is not None:row['max_abs'] = max(row['max_abs'], difference)
                if not equal:
                    row['reason'] = 'fresh batch-4 features differ; tolerance is zero'
                    break
        finally:
            canonical.unlink(missing_ok=True)
        row['regenerate_s'] = time.time()-regenerate_started
        if not row['bitwise_equal']:
            break
    passed = len(report['checks']) == len(fixtures) and all(r.get('bitwise_equal') and r.get('compared_windows') == r.get('sampled_windows') for r in report['checks'])
    cache_s = sum(r.get('cache_read_s', 0) for r in report['checks'])
    regenerate_s = sum(r.get('regenerate_s', 0) for r in report['checks'])
    report.update(status='passed' if passed else 'blocked', cache_reuse_allowed=passed,
                  cache_read_s=cache_s, regenerate_s=regenerate_s,
                  reuse_recommended=bool(passed and cache_s < regenerate_s), elapsed_s=time.time()-started)
    report['cost_rule'] = 'Reuse only after exact numerical admission AND lower aggregate measured read time on fixed training fixtures; timings are a planning sample.'
    return report
