"""Independently recompute a complete validation epoch; never fetch weights."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.cloud_io import configure_drive_environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--branch', choices=['qwen', 'openjev'], required=True)
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()
assert args.epoch >= 1
receipt = json.loads(args.receipt.read_text())
assert receipt['job'] == 'train_formal'
folder = args.receipt.parent
state = json.loads((folder / 'live_durable_status.json').read_text())
assert state['test_used'] is False
arm = state['arms'][args.branch]
record = next(v for v in arm['validation'] if v['epoch'] == args.epoch)
policy = json.loads((ROOT / 'research/training_protocol_v2.json').read_text())
manifest_path = ROOT / 'results/learning/manifest.json'
metadata_path = ROOT / 'research/formal_training_metadata.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == policy['train_manifest_sha256']
assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == policy['training_metadata_bundle_sha256']
manifest = json.loads(manifest_path.read_text())
metadata = json.loads(metadata_path.read_text())['records']
assert hashlib.sha256(json.dumps({'manifest': manifest, 'policy': policy}, sort_keys=True).encode()).hexdigest() == state['fingerprint']
validation = [r for r in manifest['records'] if r['split'] == 'val']
configure_drive_environment(ROOT)
remote = f'gdrive:remaining_20261108_v1/formal_v2/training/{args.branch}/epoch_{args.epoch}_validation.json'
command = ['rclone', 'cat', remote, '--drive-root-folder-id', 'YOUR_DRIVE_ROOT_FOLDER_ID',
           '--log-level', 'ERROR', '--contimeout', '10s', '--timeout', '15s', '--retries', '1', '--low-level-retries', '1']
response = subprocess.run(command, capture_output=True, text=True, timeout=25, check=True)
assert len(response.stdout.encode()) < 1024**2, 'Only small validation metadata belongs on the desktop'
data = json.loads(response.stdout)
rows = data['rows']
assert data['fingerprint'] == state['fingerprint']
assert len(rows) == len(validation) == record['videos'] == 163
assert [r['name'] for r in rows] == [r['name'] for r in validation]
for row, expected in zip(rows, validation):
    assert row['label'] == expected['label'] and row['windows'] == metadata[row['name']]['window_count']
    assert math.isfinite(row['max_logit']) and math.isfinite(row['topk_probability'])
assert sum(r['windows'] for r in rows) == 60114
metrics = {'video_max_ap': float(average_precision_score([r['label'] for r in rows], [r['max_logit'] for r in rows])),
           'video_max_auroc': float(roc_auc_score([r['label'] for r in rows], [r['max_logit'] for r in rows]))}
for key, value in metrics.items():
    assert abs(record[key] - value) < 1e-12
assert data['state_sha256'] == record['state_sha256']
payload_checked = None  # No independent payload reference for this epoch yet.
payload_path = folder / f'epoch{args.epoch}_checkpoint_payload_audit.json'
if payload_path.exists():
    payload = json.loads(payload_path.read_text())
    inspected = payload['arms'][args.branch]
    assert payload['test_used'] is False
    assert inspected['steps'] == args.epoch * policy['training']['steps_per_epoch']
    assert inspected['optimizer_step_min'] == inspected['optimizer_step_max'] == inspected['steps']
    assert data['state_sha256'] == inspected['current_trainables_sha256']
    cloud = payload['cloud_checkpoint_objects'][args.branch]
    assert cloud['matches_actual_Colab_checkpoint_bytes'] is True
    assert cloud['metadata']['Size'] == inspected['checkpoint_bytes']
    assert cloud['metadata']['Hashes']['md5'] == inspected['checkpoint_md5']
    payload_checked = True
stem = f'{args.branch}_epoch{args.epoch}'
complete = folder / (stem + '_validation_complete.json')
complete.write_text(response.stdout)
selection = {'fingerprint': state['fingerprint'], 'test_used': False, 'branch': args.branch, 'steps': arm['steps'],
             'validation': arm['validation'], 'best_epoch': arm['best_epoch'], 'best_ap': arm['best_ap'],
             'bad_epochs': arm['bad_epochs'], 'stopped': arm['stopped'], 'selected_artifact': arm['selected_artifact']}
selection_path = folder / (stem + '_selection_observation.json')
selection_path.write_text(json.dumps(selection, indent=2) + '\n')
report = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'branch': args.branch, 'epoch': args.epoch,
          'videos': len(rows), 'windows': sum(r['windows'] for r in rows), 'independently_recomputed_metrics': metrics,
          'match_remote_metrics_abs_tolerance': 1e-12, 'trained_state_sha256': data['state_sha256'],
          'matches_training_validation_record_state_sha256': True,
          'matches_actual_optimizer_checkpoint_payload': payload_checked,
          'input_names_labels_windows_match_frozen_validation_split': True, 'validation_file': str(complete),
          'validation_sha256': hashlib.sha256(complete.read_bytes()).hexdigest(), 'selection_observation': str(selection_path),
          'test_used': False, 'training_complete': state['status'] == 'complete', 'convergence_claimed': False,
          'limitation': 'Verifies one complete validation epoch only; training completion and plateau stopping require separate evidence.'}
(folder / (stem + '_validation_audit.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
