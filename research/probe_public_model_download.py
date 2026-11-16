"""CPU-only public, pinned model download benchmark; no Drive mount or new consent."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
started = time.time()
report = {'status': 'downloading', 'model_id': 'Qwen/Qwen3.5-2B',
          'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'started': started,
          'new_oauth_consent_needed': False}
output = Path('/content/ucf_public_download_probe.json')
output.write_text(json.dumps(report, indent=2) + '\n')
try:
    location = Path(snapshot_download(report['model_id'], revision=report['revision'], token=False,
                allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', '*.model'], max_workers=4))
    weights = []
    for path in location.glob('*.safetensors'):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                digest.update(chunk)
        weights.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    assert len(weights) == 1 and weights[0]['bytes'] == 4548221488
    report.update(status='passed', weights=weights, elapsed_s=time.time() - started, snapshot=str(location))
except Exception:
    import traceback
    report.update(status='error', error=traceback.format_exc(), elapsed_s=time.time() - started)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
