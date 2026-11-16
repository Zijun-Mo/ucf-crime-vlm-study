"""Capture native transfer errors in notebook jobs without logging credentials."""
from pathlib import Path
import re
import os
import time
import subprocess
import json
from urllib.parse import urlsplit, urlunsplit


def configure_drive_environment(root):
    """Attribute native Drive API requests to an already enabled user project."""
    path=Path(root)/'research/native_drive_policy.json'
    if not path.exists():return None
    policy=json.loads(path.read_text());project=policy['quota_project']
    assert re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]',project)
    assert policy['mode']=='request_quota_project' and policy['new_authorization_required'] is False
    os.environ['RCLONE_HEADER']='X-Goog-User-Project: '+project
    os.environ['UCF_DRIVE_QUOTA_PROJECT']=project
    return policy


def unallocated_service_failure(receipt, controller_log):
    """Only retry a proved allocation failure, never an allocated worker."""
    return bool(not receipt.get('allocated_utc') and receipt.get('allocation_requested_utc')
                and receipt.get('stop_exit_code')==0
                and receipt.get('native_drive_healthcheck',{}).get('passed_before_gpu_allocation')
                and 'Service Unavailable' in controller_log)


def sanitize(text):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(text))
    text = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '<redacted-jwt>', text)
    def url(match):
        original = match.group(0)
        parts = urlsplit(original)
        if any(word in parts.query.lower() for word in ['token', 'signature', 'sig=', 'credential', 'x-amz']):
            return urlunsplit((parts.scheme, parts.netloc, parts.path, '<redacted-query>', ''))
        return original
    return re.sub(r'https?://[^\s<>"\']+', url, text)


def transfer_timeout(command, requested):
    """Bound large local uploads by bytes and the independent runtime deadline."""
    effective = requested
    large = False
    if 'copyto' in command:
        index = command.index('copyto')
        if index + 1 < len(command):
            source = Path(command[index + 1])
            if source.is_file() and source.stat().st_size > 128 * 1024**2:
                large = True
                # Allow active chunked transfers to finish. Rclone separately
                # enforces its 60-second network-inactivity/retry limits.
                effective = max(requested, min(1200, 120 + source.stat().st_size / (2 * 1024**2)))
    deadline = os.environ.get('UCF_DEADLINE')
    if deadline is not None:
        remaining = float(deadline) - time.time() - 90
        if remaining <= 0 or (large and effective >= remaining):
            raise TimeoutError('Durable-transfer budget reserve reached before starting upload')
        if not large:
            effective = min(effective, remaining)
    return effective


def checked(command, timeout=180, log='/content/ucf_transfer_errors.log'):
    effective = transfer_timeout(command, timeout)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=effective)
    except subprocess.TimeoutExpired as error:
        output=error.output or ''
        if isinstance(output,bytes):output=output.decode(errors='replace')
        path=Path(log);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:
            stream.write(f'Transfer wall-time limit {effective:.1f}s: '+sanitize(output)[-6000:]+'\n')
        if effective < timeout:
            raise TimeoutError('Runtime budget reached while preserving a small durable result') from error
        raise
    output = sanitize(result.stdout)
    if result.returncode:
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(output + '\n')
        raise RuntimeError(f'Transfer exit={result.returncode}: {output[-3000:]}')
    return result
