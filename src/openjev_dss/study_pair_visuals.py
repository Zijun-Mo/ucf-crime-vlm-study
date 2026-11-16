"""Optional exact temporal-patch-pair reuse, admitted only by a train-fixture gate."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .runtime import decode_canonical_frames
from .study_visuals import StudyVisualStore


def _pairs(frames, valid, count):
    return [np.stack([frames[min(2 * t + j, valid)] for j in range(2)]) for t in range(count)] + [np.stack([frames[valid], frames[valid]])]


def _windows(blocks, starts, duration):
    indices = torch.as_tensor(starts)[:, None] + torch.arange(duration)[None, :]
    selected = blocks[indices.clamp(max=len(blocks) - 1)]
    return selected.reshape(len(starts), duration * blocks.shape[1], blocks.shape[2])


class GatedPairVisualStore(StudyVisualStore):
    def gate(self, fixtures, duration):
        """Require BITWISE equality; a failed gate keeps full-window extraction."""
        rows = []
        batch = 4 * duration
        for record in fixtures:
            self.guard()
            metadata = self.metadata[record['name']]
            path = self.canonical / record['name']
            starts = list(range(min(4, metadata['window_count'])))
            frames = decode_canonical_frames(path, metadata['sampled_frames'], starts, True)
            source = [np.stack([frames[min(2 * t + j, metadata['sampled_frames'])] for j in range(duration * 2)]) for t in starts]
            full = self.runtime.vision(source + [source[-1]] * (4 - len(source)))[:len(source)]
            pairs = _pairs(frames, metadata['sampled_frames'], metadata['window_count'])
            need = min(len(pairs), starts[-1] + duration + 1)
            selected = pairs[:need]
            blocks = []
            for begin in range(0, len(selected), batch):
                p = selected[begin:begin + batch]
                n = len(p)
                blocks.append(self.runtime.vision(p + [p[-1]] * (batch - n))[:n])
            blocks = torch.cat(blocks)
            reconstructed = _windows(blocks, starts, duration)
            difference = float((full.float() - reconstructed.float()).abs().max())
            rows.append({'name': record['name'], 'duration_s': duration, 'max_abs': difference, 'bitwise_equal': torch.equal(full, reconstructed)})
        return {'allowed': bool(rows) and all(r['bitwise_equal'] for r in rows), 'checks': rows,
                'gate': 'train fixtures only; equal physical token count; no tolerance relaxation'}

    def get(self, record, duration):
        if not getattr(self, 'pair_gate', {}).get('allowed'):
            return super().get(record, duration)
        metadata = self.metadata[record['name']]
        assert metadata['source_sha256'] == record['sha256']
        signature = {**self.signature, 'representation': 'temporal_patch_pairs', 'pair_batch': 4 * duration,
                     'duration_gate': duration, 'canonical_sha256': metadata['canonical_sha256'],
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
            blocks = entry['features']
            assert blocks.shape[:2] == (metadata['window_count'] + 1, 49) and torch.isfinite(blocks).all()
        else:
            self.guard()
            local = self.local_cache / ('canonical_' + record['name'])
            shutil.copyfile(self.canonical / record['name'], local)
            assert hashlib.sha256(local.read_bytes()).hexdigest() == metadata['canonical_sha256']
            try:
                frames = decode_canonical_frames(local, metadata['sampled_frames'], [0], True)
                pairs = _pairs(frames, metadata['sampled_frames'], metadata['window_count'])
                batch, blocks = 4 * duration, []
                with torch.no_grad():
                    for begin in range(0, len(pairs), batch):
                        self.guard()
                        selected = pairs[begin:begin + batch]
                        n = len(selected)
                        blocks.append(self.runtime.vision(selected + [selected[-1]] * (batch - n))[:n])
                blocks = torch.cat(blocks)
                assert torch.isfinite(blocks).all()
                temporary = target.with_suffix('.partial')
                torch.save({'signature': signature, 'features': blocks}, temporary)
                temporary.replace(target)
                self.sync(target)
            finally:
                local.unlink(missing_ok=True)
        return _windows(blocks, list(range(metadata['window_count'])), duration)
