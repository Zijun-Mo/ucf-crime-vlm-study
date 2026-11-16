"""Read the last cloud-committed training state without fetching any weights."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.cloud_io import configure_drive_environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--validation-details', action='store_true', help='Read current-epoch partial validation JSONs and verify exact video/window coverage')
args = parser.parse_args()
receipt = json.loads(args.receipt.read_text())
assert receipt['job'] == 'train_formal'
configure_drive_environment(ROOT)
command = ['rclone', 'cat', 'gdrive:remaining_20261108_v1/formal_v2/training/status.json',
           '--drive-root-folder-id', 'YOUR_DRIVE_ROOT_FOLDER_ID', '--log-level', 'ERROR',
           '--contimeout', '10s', '--timeout', '15s', '--retries', '1', '--low-level-retries', '1']
result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=True)
data = json.loads(result.stdout)
assert data['test_used'] is False and set(data['arms']) == {'qwen', 'openjev'}
for arm in data['arms'].values():
    assert arm['steps'] == len(arm['rows'])
path = args.receipt.parent / 'live_durable_status.json'
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n')
temporary.replace(path)
summary = {k: data.get(k) for k in ['status', 'stage', 'epoch_cursor', 'step_cursor', 'generation', 'elapsed_s', 'validation_progress']}
summary.update(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               evidence='Last cloud-committed status; observe the actual worker separately for current liveness.',
               arms={name: {'steps': arm['steps'], 'validation_epochs': len(arm['validation']),
                   'stopped': arm['stopped'], 'best_epoch': arm.get('best_epoch'), 'best_ap': arm.get('best_ap'),
                   'last_training_row': {k: arm['rows'][-1][k] for k in ['step', 'epoch', 'loss', 'grad_norm', 'seconds']} if arm['rows'] else None}
                   for name, arm in data['arms'].items()})
if args.validation_details:
    manifest_path = ROOT / 'results/learning/manifest.json'
    metadata_path = ROOT / 'research/formal_training_metadata.json'
    policy = json.loads((ROOT / 'research/training_protocol_v2.json').read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == policy['train_manifest_sha256']
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == policy['training_metadata_bundle_sha256']
    validation = [r for r in json.loads(manifest_path.read_text())['records'] if r['split'] == 'val']
    metadata = json.loads(metadata_path.read_text())['records']
    total_windows = sum(metadata[r['name']]['window_count'] for r in validation)
    epoch = data['epoch_cursor'] + 1
    def details(name):
        remote = f'gdrive:remaining_20261108_v1/formal_v2/training/{name}/epoch_{epoch}_validation.json'
        requested = command.copy()
        requested[2] = remote
        try:
            response = subprocess.run(requested, capture_output=True, text=True, timeout=25)
            if response.returncode:
                # https://rclone.org/docs/#list-of-exit-codes: missing paths
                # can return3/4 with no stderr under --log-level ERROR.
                missing = response.returncode in (3, 4) or any(term in response.stderr.lower() for term in ['object not found', 'directory not found'])
                return name, {'epoch': epoch, 'partial_file_available': False,
                    'native_exit_code': response.returncode,
                    'observation': 'not yet published' if missing else 'native read failed; progress unproven'}
            partial = json.loads(response.stdout)
            rows = partial['rows']
            assert partial['fingerprint'] == data['fingerprint']
            assert [r['name'] for r in rows] == [r['name'] for r in validation[:len(rows)]]
            for row, record in zip(rows, validation):
                assert row['label'] == record['label'] and row['windows'] == metadata[row['name']]['window_count']
                assert math.isfinite(row['max_logit']) and math.isfinite(row['topk_probability'])
            target = args.receipt.parent / f'{name}_epoch_{epoch}_validation_live.json'
            temporary = target.with_suffix('.tmp')
            temporary.write_text(response.stdout)
            temporary.replace(target)
            windows = sum(r['windows'] for r in rows)
            next_record = validation[len(rows)] if len(rows) < len(validation) else None
            return name, {'epoch': epoch, 'partial_file_available': True, 'videos': len(rows),
                'total_videos': len(validation), 'windows': windows, 'total_windows': total_windows,
                'window_fraction': windows / total_windows, 'source': remote,
                'next_video': next_record['name'] if next_record else None,
                'next_video_windows': metadata[next_record['name']]['window_count'] if next_record else None,
                'ordered_video_labels_and_window_counts_verified': True,
                'complete_video_coverage': len(rows) == len(validation),
                'note': 'Counts fully scored videos only; the next video may be in progress. This cloud file can precede the committed status pointer. No partial AP is used for selection.'}
        except subprocess.TimeoutExpired:
            return name, {'epoch': epoch, 'observation': 'native read timeout; progress unproven'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        summary['current_epoch_validation_files'] = dict(pool.map(details, data['arms']))
print(json.dumps(summary, indent=2))
