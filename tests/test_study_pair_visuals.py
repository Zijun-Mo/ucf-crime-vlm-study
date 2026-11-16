import hashlib
import cv2
import numpy as np
import torch
import importlib.metadata
from openjev_dss.study_pair_visuals import GatedPairVisualStore


class PairEncoder:
    def __init__(self, temporal_context=False):
        self.temporal_context = temporal_context
        self.calls = 0

    def vision(self, windows):
        self.calls += 1
        output = []
        for w in windows:
            features = []
            for i in range(0, len(w), 2):
                value = float(w[i:i + 2].astype(np.float32).mean())
                features.append(torch.full((49, 2), value))
            result = torch.cat(features)
            if self.temporal_context:
                result += len(w)
            output.append(result)
        return torch.stack(output)


def fixture(tmp_path):
    directory = tmp_path / 'canonical'
    directory.mkdir()
    path = directory / 'fixture.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 2, (224, 224))
    for i in range(8):
        writer.write(np.full((224, 224, 3), i * 20, np.uint8))
    writer.release()
    record = {'name': path.name, 'n_frames': 105, 'fps': 30, 'sha256': 'frozen-source'}
    metadata = {path.name: {'source_sha256': record['sha256'], 'sampled_frames': 7,
                           'window_count': 4, 'canonical_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    return directory, record, metadata


def test_gate_rejects_temporal_context_and_accepts_exact_pair_cache(tmp_path, monkeypatch):
    actual_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'test-encoder' if name == 'transformers' else actual_version(name))
    directory, record, metadata = fixture(tmp_path)
    wrong = GatedPairVisualStore(PairEncoder(True), directory, metadata, tmp_path / 'wrong', 'test')
    assert not wrong.gate([record], 4)['allowed']
    encoder = PairEncoder()
    good = GatedPairVisualStore(encoder, directory, metadata, tmp_path / 'correct', 'test')
    good.pair_gate = good.gate([record], 4)
    assert good.pair_gate['allowed']
    expected = good.get(record, 4)
    assert expected.shape == (4, 196, 2)
    calls = encoder.calls
    cached = good.get(record, 4)
    assert encoder.calls == calls
    torch.testing.assert_close(expected, cached, rtol=0, atol=0)
