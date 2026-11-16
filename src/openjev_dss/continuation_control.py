"""Explicit execution-only extensions of a frozen training trajectory."""
import hashlib
import json
from pathlib import Path


def load_continuation(root, path, policy_path):
    if not path:
        return None
    root = Path(root)
    control = json.loads((root / path).read_text())
    assert control['authorized'] is True
    assert control['base_policy'] == str(policy_path)
    assert hashlib.sha256((root / policy_path).read_bytes()).hexdigest() == control['base_policy_sha256']
    assert control['until_validation_plateau'] is True
    assert control['additional_training_limit_cu'] == 20.0
    assert control['gpu_allowlist'] == ['L4']
    assert control['test_during_training'] is False
    assert control['output_remote'] != control['resume_remote']
    if control.get('resume_requires_cache_bundle'):
        bundle=control.get('visual_cache_bundle')
        assert bundle and bundle.get('manifest_sha256') and bundle.get('inventory_sha256'), 'Verified CPU cache bundle required before allocating training GPU'
    return control


def verify_resume(control, state, saved):
    """Fail closed before training if the durable cursor/optimizer was lost."""
    assert state['fingerprint'] == control['base_fingerprint'] == saved['fingerprint']
    assert state['steps'] == saved['steps'] >= control['resume_steps']
    assert state['epoch_cursor'] >= control['completed_epochs']
    counters = [int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
    assert counters and all(v == saved['steps'] for v in counters)
    if state['steps'] == control['resume_steps']:
        assert state['epoch_cursor'] == control['completed_epochs']
        assert state['video_cursor'] == control['resume_video_cursor']
    return counters
