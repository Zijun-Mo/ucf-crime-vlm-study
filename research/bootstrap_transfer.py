"""Install a pinned rclone and verify the EXISTING task-scoped config on Colab.

Credentials are supplied as a separate file by `colab upload`, never in this code.
"""
import hashlib
import configparser
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

config = Path('/content/ucf-rclone.conf')
if not config.exists():
    raise RuntimeError('Upload the existing rclone config before running this preflight')
config.chmod(0o600)
binary = Path('/content/ucf-bin/rclone')
expected = 'ed7ab1f17876bb006c3df3949a3044959671afd9112362dacaa5dd1647735bc1'
if not binary.exists():
    binary.parent.mkdir(exist_ok=True)
    archive = Path('/content/ucf-rclone.zip')
    if not archive.exists():
        with zipfile.ZipFile('/content/ucf_project.zip') as project:
            asset = json.loads(project.read('research/bootstrap_assets.json'))
        parser = configparser.ConfigParser()
        parser.read(config)
        access = json.loads(parser['gdrive']['token'])['access_token']
        headers={'Authorization': 'Bearer ' + access}
        if os.environ.get('UCF_DRIVE_QUOTA_PROJECT'):
            headers['X-Goog-User-Project']=os.environ['UCF_DRIVE_QUOTA_PROJECT']
        request = urllib.request.Request(
            'https://www.googleapis.com/drive/v3/files/' + asset['rclone_zip_drive_id'] + '?alt=media',
            headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as target:
            shutil.copyfileobj(response, target)
        if archive.stat().st_size != asset['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest() != asset['sha256']:
            raise RuntimeError('Bootstrap archive digest mismatch')
    with zipfile.ZipFile(archive) as z:
        with z.open('rclone-v1.75.2-linux-amd64/rclone') as source, binary.open('wb') as target:
            shutil.copyfileobj(source, target)
    archive.unlink()
if hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Unexpected rclone binary digest')
binary.chmod(0o700)
base = [str(binary), '--config', str(config), '--contimeout', '15s', '--timeout', '30s', '--retries', '2', '--low-level-retries', '2']
result = subprocess.run(base + ['lsf', 'gdrive:', '--max-depth', '1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
if result.returncode:
    safe = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '<redacted-jwt>', result.stdout)
    raise RuntimeError('Native rclone verification failed: ' + safe[-3000:])
out = result.stdout
report = {'rclone_version': '1.75.2', 'binary_sha256': expected,
          'project_folders': out.splitlines(), 'no_interactive_authorization': True,
          'fuse_device': Path('/dev/fuse').exists(), 'fusermount': shutil.which('fusermount') or shutil.which('fusermount3')}
assert 'formal_stride1/' in report['project_folders']
Path('/content/ucf_transfer_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
