"""Bounded Colab CLI preflight with unconditional VM cleanup and receipt verification."""
import datetime
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
session = 'ucf-bf16-probe'
log_path = ROOT / 'research/gpu_preflight_controller.log'
receipt = {'session': session, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'requested_gpu': 'L4', 'hard_stop_seconds': 900, 'status': 'preparing'}
local_report = ROOT / 'research/gpu_preflight_result.json'

def command(args, timeout=180, check=True, capture=False):
    with log_path.open('a') as log:
        log.write('\nCOMMAND ' + ' '.join(map(str, args)) + '\n')
        log.flush()
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        result.stdout = re.sub(r'(colab-runtime-proxy-token=)[^\s&]+', r'\1<redacted>', result.stdout)
        result.stdout = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '<redacted-jwt>', result.stdout)
        log.write(result.stdout)
    if check and result.returncode:
        raise RuntimeError(f'Colab command failed: {args[:3]}, exit={result.returncode}')
    return result

try:
    secret = Path((ROOT / '.colab-work/credential_config_path').read_text().strip())
    assert secret.is_file() and (ROOT / '.colab-work/fusermount3').is_file()
    # Reuse the existing authorization; copy only the task remote and fresh token.
    source = configparser.ConfigParser()
    source.read(Path.home() / '.config/rclone/rclone.conf')
    scoped = configparser.ConfigParser()
    scoped['gdrive'] = dict(source['gdrive'])
    scoped['gdrive']['root_folder_id'] = 'YOUR_DRIVE_ROOT_FOLDER_ID'
    with secret.open('w') as target:
        scoped.write(target)
    secret.chmod(0o600)
    archive = ROOT / '.colab-work/ucf_project.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ['src', 'research']:
            for p in (ROOT / folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py', '.json']:
                    z.write(p, p.relative_to(ROOT).as_posix())
    receipt['source_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt['usage_before'] = command(['colab', 'usage'], capture=True).stdout
    command(['colab', 'new', '-s', session, '--gpu', 'L4'])
    command(['colab', 'exec', '-s', session, '-f', 'research/gpu_watchdog.py', '--timeout', '30'], timeout=60)
    for local, remote in [(secret, '/content/ucf-rclone.conf'), (archive, '/content/ucf_project.zip'),
                          (ROOT / '.colab-work/fusermount3', '/content/ucf-fusermount3')]:
        command(['colab', 'upload', '-s', session, str(local), remote], timeout=180)
    command(['colab', 'exec', '-s', session, '-f', 'research/bootstrap_transfer.py', '--timeout', '120'], timeout=160)
    helper = ROOT / '.colab-work/prepare_fuse_helper.py'
    helper.write_text("import os\nfrom pathlib import Path\nPath('/content/ucf-fusermount3').replace('/content/ucf-bin/fusermount3')\nPath('/content/ucf-bin/fusermount3').chmod(0o755)\nos.environ['PATH']='/content/ucf-bin:'+os.environ['PATH']\n")
    command(['colab', 'exec', '-s', session, '-f', str(helper), '--timeout', '20'], timeout=40)
    command(['colab', 'exec', '-s', session, '-f', 'research/mount_rclone.py', '--timeout', '90'], timeout=120)
    mount_receipt = ROOT / 'research/gpu_mount_verified.json'
    if mount_receipt.exists():
        mount_receipt.unlink()
    command(['colab', 'download', '-s', session, '/content/ucf_mount_verified.json', str(mount_receipt)], timeout=60)
    assert json.loads(mount_receipt.read_text())['test_videos'] == 290
    receipt['usage_allocated'] = command(['colab', 'usage'], capture=True).stdout
    receipt['status'] = 'probing'
    (ROOT / 'research/gpu_preflight_controller.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('GPU preflight launched with remote 900-second watchdog', flush=True)
    command(['colab', 'exec', '-s', session, '-f', 'research/gpu_preflight.py', '--timeout', '720'], timeout=750)
    command(['colab', 'download', '-s', session, '/content/ucf_gpu_preflight.json', str(local_report)])
    result = json.loads(local_report.read_text())
    if result['status'] != 'passed':
        raise RuntimeError('GPU numerical preflight failed; inspect research/gpu_preflight_result.json')
    receipt['status'] = 'passed'
except Exception as error:
    receipt.update(status='error', error=str(error))
finally:
    try:
        stopped = command(['colab', 'stop', '-s', session], timeout=90, check=False, capture=True)
        receipt['stop_output'] = stopped.stdout
        receipt['stop_exit_code'] = stopped.returncode
        receipt['usage_after'] = command(['colab', 'usage'], capture=True).stdout
    except Exception as error:
        receipt['cleanup_error'] = str(error)
    receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT / 'research/gpu_preflight_controller.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
sys.exit(0 if receipt['status'] == 'passed' else 1)
