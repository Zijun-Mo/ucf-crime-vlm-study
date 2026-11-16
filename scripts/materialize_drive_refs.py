"""Internal transfer helper: accepts ephemeral authenticated refs on stdin.

Never persists or prints download URLs. Durable provenance is generated separately.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'archive' / 'drive'

def download(item):
    path = ROOT / item['drive_path']
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Path escapes archive')
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item['size'])
    if path.exists() and path.stat().st_size == expected:
        return {'path': item['drive_path'], 'status': 'present', 'bytes': expected}
    temp = path.with_name(path.name + '.partial')
    try:
        with urllib.request.urlopen(item['download_url'], timeout=90) as response, temp.open('wb') as target:
            shutil.copyfileobj(response, target)
        actual = temp.stat().st_size
        if actual != expected:
            raise ValueError(f'Size mismatch: expected {expected}, got {actual}')
        os.replace(temp, path)
        return {'path': item['drive_path'], 'status': 'downloaded', 'bytes': actual}
    except Exception as error:
        return {'path': item['drive_path'], 'status': 'error', 'type': type(error).__name__}

if __name__ == '__main__':
    items = json.load(sys.stdin)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(download, items))
    print(json.dumps(results, ensure_ascii=False))
    sys.exit(any(row['status'] == 'error' for row in results))
