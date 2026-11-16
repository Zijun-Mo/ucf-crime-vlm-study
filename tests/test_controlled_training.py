import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from openjev_dss import controlled_training as training


class TinyModel(torch.nn.Linear):
    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True)
        torch.save(self.state_dict(), Path(path) / 'weights.pt')


def test_resuming_after_budget_stop_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state: None)
    normal = {'name': 'normal', 'label': 0, 'split': 'train'}
    abnormal = {'name': 'abnormal', 'label': 1, 'split': 'train'}
    validation = [{'name': str(i), 'label': i % 2, 'split': 'val'} for i in range(163)]
    manifest = {'records': [normal, abnormal] + validation, 'audit': {'near_duplicate_visual_audit': 'passed'}}
    plan = [(0, [(normal, abnormal)] * 4) for _ in range(160)]
    monkeypatch.setattr(training, 'schedule', lambda records, epochs: plan)
    torch.manual_seed(12)
    initial = TinyModel(2, 1)

    def runtime():
        model = copy.deepcopy(initial)
        return SimpleNamespace(model=model, full=model, head=torch.nn.Identity(),
                               forward=lambda x: model(x).flatten())

    def provider(record, epoch, exhaustive):
        sign = 1. if record['label'] else -1.
        return torch.tensor([[sign, .2], [sign * .5, -.3]])

    complete = training.run_single_mil(runtime(), manifest, provider, tmp_path / 'direct', time.time() + 1000)
    assert complete['status'] == 'complete'
    first = True

    def stop_once_after_durable_save():
        nonlocal first
        if first:
            first = False
            raise TimeoutError('simulated budget boundary')

    partial = training.run_single_mil(runtime(), manifest, provider, tmp_path / 'resumed', time.time() + 1000,
                                      sync=stop_once_after_durable_save)
    assert partial['status'] == 'budget_stopped' and partial['steps'] == 10
    resumed = training.run_single_mil(runtime(), manifest, provider, tmp_path / 'resumed', time.time() + 1000)
    assert resumed['status'] == 'complete' and resumed['steps'] == 160 and resumed['validation_videos'] == 163
    assert resumed['final_lora_sha256'] == complete['final_lora_sha256']
    assert not resumed['test_used']
    assert len(resumed['rows']) == 160


def test_two_publish_failures_cannot_overwrite_last_durable_checkpoint(tmp_path, monkeypatch):
    import shutil
    import pytest
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state: None)
    normal = {'name': 'normal', 'label': 0, 'split': 'train'}
    abnormal = {'name': 'abnormal', 'label': 1, 'split': 'train'}
    manifest = {'records': [normal, abnormal], 'audit': {'near_duplicate_visual_audit': 'passed'}}
    plan = [(0, [(normal, abnormal)] * 4) for _ in range(160)]
    monkeypatch.setattr(training, 'schedule', lambda records, epochs: plan)
    model = TinyModel(2, 1)
    runtime = SimpleNamespace(model=model, full=model, head=torch.nn.Identity(),
                              forward=lambda x: model(x).flatten())
    local, cloud = tmp_path/'local', tmp_path/'cloud'
    cloud.mkdir()
    calls = 0

    def sync():
        nonlocal calls
        calls += 1
        for checkpoint in local.glob('resume_*.pt'):
            shutil.copyfile(checkpoint, cloud/checkpoint.name)
        if calls >= 2:
            raise TimeoutError('checkpoint data arrived, but status commit failed')
        shutil.copyfile(local/'status.json', cloud/'status.json')

    def provider(record, epoch, exhaustive):
        sign = 1. if record['label'] else -1.
        return torch.tensor([[sign, .2], [sign * .5, -.3]])

    with pytest.raises(TimeoutError):
        training.run_single_mil(runtime, manifest, provider, local, time.time()+1000, sync)
    durable = json.loads((cloud/'status.json').read_text())
    checkpoint = torch.load(cloud/f"resume_{durable['generation']%2}.pt", map_location='cpu', weights_only=True)
    assert durable['steps'] == checkpoint['steps'] == 10
    assert calls == 3  # step 10 committed; step 20 and final retry both interrupted.
