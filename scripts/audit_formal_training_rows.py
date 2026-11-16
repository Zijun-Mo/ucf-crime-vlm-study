"""Audit recorded formal optimizer inputs on CPU; do not read any weights."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.learning import schedule


def audit(status_path, contract_path):
    raw = status_path.read_bytes()
    state = json.loads(raw)
    contract = json.loads(contract_path.read_text())
    policy_path = ROOT / 'research/training_protocol_v2.json'
    policy = json.loads(policy_path.read_text())
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == contract['policy_sha256']
    for name, digest in contract['source'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    manifest_path = ROOT / 'results/learning/manifest.json'
    metadata_path = ROOT / 'research/formal_training_metadata.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == policy['train_manifest_sha256']
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == policy['training_metadata_bundle_sha256']
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())['records']
    assert state['test_used'] is False and set(state['arms']) == {'qwen', 'openjev'}
    assert hashlib.sha256(json.dumps({'manifest': manifest, 'policy': policy}, sort_keys=True).encode()).hexdigest() == state['fingerprint']
    planned = schedule(manifest['records'], state['epoch_limit'])
    assert len(planned) == state['epoch_limit'] * policy['training']['steps_per_epoch']
    physical = policy['training']['physical_lm_batch']
    arms = {}
    for branch, arm in state['arms'].items():
        rows = arm['rows']
        assert arm['steps'] == len(rows)
        names = set()
        for index, row in enumerate(rows):
            epoch, pairs = planned[index]
            expected = [record for pair in pairs for record in pair]
            assert row['step'] == index + 1 and row['epoch'] == epoch + 1
            assert len(expected) == policy['training']['bags_per_step']
            assert all(record['split'] == 'train' for record in expected)
            assert [record['label'] for record in expected] == [0, 1] * len(pairs)
            assert row['videos'] == [record['name'] for record in expected]
            counts = [min(metadata[record['name']]['window_count'], policy['training']['windows_per_bag_max']) for record in expected]
            assert row['bag_windows'] == counts
            assert all(math.isfinite(row[key]) for key in ['loss', 'grad_norm', 'seconds'])
            assert row['grad_norm'] >= 0 and row['seconds'] >= 0
            info = row['stats']
            assert info['score_batch'] == info['backward_batch'] == physical
            assert info['original_batch_positions_preserved'] is True
            assert info['score_forward_grad_enabled'] is True
            assert math.isfinite(info['max_recomputed_score_abs']) and info['max_recomputed_score_abs'] >= 0
            assert info['ss'] == 0 and len(info['bags']) == len(counts)
            assert info['score_calls'] == math.ceil(sum(counts) / physical)
            assert 0 <= info['backward_calls'] <= info['score_calls']
            assert info['backward_window_slots'] == info['backward_calls'] * physical
            assert 0 <= info['active_windows'] <= min(sum(counts), info['backward_window_slots'])
            for bag, count in zip(info['bags'], counts):
                assert bag['n'] == count and bag['k'] == math.ceil(count / 8)
                assert bag['suppressed'] == 0 and bag['ss_top'] == bag['ss_bottom'] == []
                assert len(bag['mil_top']) == bag['k'] == len(set(bag['mil_top']))
                assert all(0 <= position < count for position in bag['mil_top'])
            names.update(row['videos'])
        epoch_rows = {epoch: [row for row in rows if row['epoch'] == epoch] for epoch in sorted({row['epoch'] for row in rows})}
        arms[branch] = {
            'steps_checked': len(rows), 'unique_training_videos': len(names),
            'recorded_rows_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            'epochs': {str(epoch): {'steps_checked': len(part), 'full_training_epoch_recorded': len(part) == policy['training']['steps_per_epoch'],
                'mean_training_loss': statistics.mean(row['loss'] for row in part),
                'mean_recorded_model_step_seconds': statistics.mean(row['seconds'] for row in part)} for epoch, part in epoch_rows.items()},
            'max_recorded_recomputed_score_abs': max((row['stats']['max_recomputed_score_abs'] for row in rows), default=None)}
    common = min(arm['steps'] for arm in state['arms'].values())
    for index in range(common):
        first, second = (state['arms'][branch]['rows'][index] for branch in ['qwen', 'openjev'])
        assert first['videos'] == second['videos'] and first['bag_windows'] == second['bag_windows']
    return {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status_source': str(status_path), 'status_source_sha256_at_observation': hashlib.sha256(raw).hexdigest(),
        'status_source_may_be_mutable': True, 'state_fingerprint': state['fingerprint'],
        'policy_and_contract_source_hashes_match': True, 'contract_sources_checked': len(contract['source']),
        'recorded_inputs_match_frozen_train_schedule': True, 'common_paired_updates_checked': common,
        'recorded_physical_batches_and_gradient_diagnostics_consistent': True,
        'all_observed_losses_gradient_norms_and_timings_finite': True, 'arms': arms,
        'test_used': False, 'convergence_claimed': False,
        'scope': 'Recorded optimizer rows only. Does not inspect tensors, checkpoint payloads, validation scores or prove convergence; later status files can reproduce the same recorded row prefixes.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    assert receipt['job'] == 'train_formal'
    folder = args.receipt.parent
    source = folder / 'result.json'
    if not source.exists():
        source = folder / 'live_durable_status.json'
    report = audit(source, folder / 'contract.json')
    counts = '_'.join(str(report['arms'][branch]['steps_checked']) for branch in ['qwen', 'openjev'])
    destination = folder / f'training_rows_{counts}_audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
