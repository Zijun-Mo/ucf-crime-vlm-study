"""Canonical video windows and prompt-independent frozen visual feature caches."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
import cv2


def duration_windows(path, valid, starts, duration=4, batch=4):
    """Sequential decode with exactly the historical last-frame padding rule."""
    import cv2
    if duration not in [2, 4, 8] or batch < 1 or not starts or starts != sorted(starts):
        raise ValueError('Invalid frozen temporal configuration')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open canonical video')
    cursor, frames = 0, {}
    try:
        for begin in range(0, len(starts), batch):
            selected = starts[begin:begin + batch]
            first = min(2 * selected[0], valid)
            wanted = min(2 * selected[-1] + duration * 2 - 1, valid)
            while cursor <= wanted:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError('Canonical video ended before its recorded padding frame')
                if cursor >= first:
                    frames[cursor] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cursor += 1
            yield begin, [np.stack([frames[min(2 * t + j, valid)] for j in range(duration * 2)]) for t in selected]
            if begin + batch < len(starts):
                cut = min(2 * starts[begin + batch], valid)
                frames = {k: value for k, value in frames.items() if k >= cut}
        assert cursor == valid + 1 and not cap.read()[0]
    finally:
        cap.release()


class StudyVisualStore:
    """Cache entries are keyed by source, canonical bytes, revision and environment.

    `cloud_cache` is a read-only rclone mount of previous entries; new entries go
    to SSD and a caller-controlled sync callback. No original files are modified.
    """
    def __init__(self, runtime, canonical, metadata, local_cache, revision,
                 cloud_cache=None, sync=lambda path: None, guard=lambda: None):
        self.runtime, self.canonical = runtime, Path(canonical)
        self.metadata = metadata
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.cloud_cache = Path(cloud_cache) if cloud_cache else None
        self.sync, self.guard = sync, guard
        self.signature = {'model_revision': revision, 'dtype': 'bf16', 'vision_batch': 4,
                          'torch': str(torch.__version__), 'transformers': importlib.metadata.version('transformers'),
                          'opencv': cv2.__version__, 'size': 224, 'fps': 2,
                          'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu-test',
                          'cuda_capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}

    def get(self, record, duration):
        metadata = self.metadata[record['name']]
        assert metadata['source_sha256'] == record['sha256']
        signature = {**self.signature, 'duration': duration, 'canonical_sha256': metadata['canonical_sha256'],
                     'source_sha256': record['sha256']}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        target = self.local_cache / (digest + '.pt')
        if not target.exists() and self.cloud_cache and (self.cloud_cache / target.name).exists():
            self.guard()
            temporary = target.with_suffix('.partial')
            shutil.copyfile(self.cloud_cache / target.name, temporary)
            temporary.replace(target)
        if target.exists():
            entry = torch.load(target, map_location='cpu', weights_only=True)
            assert entry['signature'] == signature
            features = entry['features']
            assert features.shape[:2] == (metadata['window_count'], duration * 49)
            assert torch.isfinite(features).all()
            return features
        self.guard()
        source = self.canonical / record['name']
        local = self.local_cache / ('canonical_' + record['name'])
        shutil.copyfile(source, local)
        assert hashlib.sha256(local.read_bytes()).hexdigest() == metadata['canonical_sha256']
        starts = list(range(metadata['window_count']))
        assert len(starts) == math.ceil(record['n_frames'] / record['fps'])
        features = None
        try:
            with torch.no_grad():
                for offset, windows in duration_windows(local, metadata['sampled_frames'], starts, duration, 4):
                    self.guard()
                    actual = len(windows)
                    # Equal physical batch for all methods and every video tail.
                    windows += [windows[-1]] * (4 - actual)
                    values = self.runtime.vision(windows)[:actual]
                    if features is None:
                        features = torch.empty((len(starts), *values.shape[1:]), dtype=values.dtype)
                    features[offset:offset + actual] = values
            assert features is not None and torch.isfinite(features).all()
            temporary = target.with_suffix('.partial')
            torch.save({'signature': signature, 'features': features}, temporary)
            temporary.replace(target)
            self.sync(target)
            return features
        finally:
            local.unlink(missing_ok=True)
