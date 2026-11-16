"""Read-only, noninteractive FUSE access using the already-authorized rclone remote."""
import json
import os
from pathlib import Path
import subprocess
import time

mount = Path('/content/ucf_drive')
mount.mkdir(exist_ok=True)
command = ['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf',
           'mount', 'gdrive:', str(mount), '--read-only', '--daemon', '--daemon-wait', '30s',
           '--vfs-cache-mode', 'full', '--vfs-cache-max-size', '100G', '--vfs-cache-max-age', '2h',
           '--dir-cache-time', '1h', '--poll-interval', '0', '--buffer-size', '16M',
           '--vfs-read-chunk-size', '16M', '--vfs-read-chunk-size-limit', '128M',
           '--contimeout', '15s', '--timeout', '30s', '--retries', '2', '--low-level-retries', '2',
           '--log-level', 'ERROR', '--log-file', '/content/ucf-rclone-mount.log']
if not os.path.ismount(mount):
    subprocess.run(command, check=True, timeout=45)
for attempt in range(15):
    try:
        # Let a freshly mounted FUSE directory finish its first remote listing.
        list(mount.iterdir())
        project_config = json.loads((mount / 'formal_stride1/config.json').read_text())
        break
    except FileNotFoundError:
        if attempt == 14:
            raise
        time.sleep(2)
train_manifest = json.loads((mount / 'weak_dss_learning_10cu_v1/manifest.json').read_text())
report = {'mount': str(mount), 'read_only': True, 'no_new_oauth_consent': True,
          'test_videos': project_config['manifest']['videos'],
          'train_counts': train_manifest['counts'],
          'template_bytes': (mount / 'formal_stride1/template.pt').stat().st_size,
          'best_state_bytes': (mount / 'weak_dss_learning_10cu_v1/training/A/best_state.pt').stat().st_size}
assert report['test_videos'] == 290
receipt=Path('/content/ucf_mount_verified.json')
temporary=receipt.with_suffix('.tmp')
temporary.write_text(json.dumps(report, indent=2) + '\n')
temporary.replace(receipt)
print(json.dumps(report, indent=2))
