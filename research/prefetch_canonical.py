"""Warm read-only FUSE bytes ahead of a live worker; never decode or run a model."""
import concurrent.futures
import json
from pathlib import Path
import time

root = Path('/content/ucf_drive')
handle = json.loads(Path('/content/ucf_background_handle.json').read_text())
records = json.loads((root / 'formal_stride1/manifest.json').read_text())['videos']
destination = Path('/content/ucf_prefetch_status.json')

def alive():
    try:
        fields = Path(f"/proc/{handle['pid']}/stat").read_text().split(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == handle['start_ticks']
    except FileNotFoundError:
        return False


def warm(record):
    count = 0
    with (root / 'formal_stride1/canonical' / record['name']).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            count += len(block)
    return count


seen, pending, warmed, total = set(), {}, 0, 0
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    while alive():
        for name, future in list(pending.items()):
            if future.done():
                try:
                    total += future.result()
                    warmed += 1
                except Exception:
                    seen.discard(name)
                del pending[name]
        predictions = Path('/content/ucf_results/qwen_zs_p1_4s/predictions')
        completed = {p.name.removesuffix('.json') for p in predictions.glob('*.json')}
        remaining = [r for r in records if r['name'] not in completed]
        # Bound ahead-read I/O so it does not flood shared Drive quotas.
        for record in remaining[:4]:
            if record['name'] not in seen and len(pending) < 2:
                pending[record['name']] = pool.submit(warm, record)
                seen.add(record['name'])
        destination.write_text(json.dumps({'warmed': warmed, 'bytes': total, 'pending': list(pending), 'updated': time.time()}))
        if not remaining:
            break
        time.sleep(2)
