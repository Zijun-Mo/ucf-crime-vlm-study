"""Foreground capacity polling: retry only proved unallocated 503 failures."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.cloud_io import unallocated_service_failure


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator', type=int, required=True)
    parser.add_argument('--retry-delay', type=int, default=300)
    args = parser.parse_args()
    assert args.retry_delay >= 120
    lock = (ROOT / '.colab-work/capacity_poll.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid, start = args.coordinator, identity(args.coordinator)
    assert start, 'The coordinator to adopt must be live'
    report = {'pid': __import__('os').getpid(), 'restarts': [],
              'policy': 'Wait for actual kernel connectivity; only retry proved pre-allocation 503 failures. Stop on application errors. Keep 9B last.'}
    last_print = 0

    def save(status, **details):
        nonlocal last_print
        report.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      coordinator_pid=pid, coordinator_start_ticks=start, **details)
        path = ROOT / 'research/capacity_poll_state.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(path)
        if time.monotonic() - last_print >= 30 or status in ['connected', 'needs_inspection']:
            print(json.dumps({k: report.get(k) for k in ['updated_utc', 'status', 'coordinator_pid', 'active_receipt', 'retry_in_s', 'reason'] if k in report}), flush=True)
            last_print = time.monotonic()

    while True:
        queue = json.loads((ROOT / 'research/queue_state.json').read_text())
        receipt_path = ROOT / queue.get('active_receipt', queue['adopted_receipt'])
        receipt = json.loads(receipt_path.read_text())
        save('waiting_for_capacity', active_receipt=str(receipt_path.relative_to(ROOT)))
        if receipt.get('allocated_utc') and not receipt.get('finished_utc'):
            interpreter = '/opt/colab-cli/bin/python'
            command = [interpreter, 'scripts/inspect_colab_job.py', receipt['session']]
            if receipt.get('endpoint'):
                command += ['--endpoint', receipt['endpoint']]
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=35, check=True)
                observed = json.loads(result.stdout)
                if observed.get('assignment_present') and observed.get('kernels_http_status') == 200 and observed.get('kernels'):
                    save('connected', connection=observed, retry_in_s=None,
                         next='Existing coordinator continues the formal CUDA gate and training; no duplicate runtime.')
                    return
            except (subprocess.SubprocessError, ValueError):
                # Observation failure is not a terminal runtime observation.
                pass
        if identity(pid) == start:
            time.sleep(15)
            continue
        # Never replace a missing coordinator while its actual controller lives.
        if identity(receipt['controller_pid']) or not receipt.get('finished_utc'):
            save('needs_inspection', reason='Coordinator ended without a verified terminal allocation controller; preserve the existing request/runtime.')
            return
        log = receipt_path.parent.joinpath('controller.log').read_text(errors='replace')
        if not unallocated_service_failure(receipt, log):
            save('needs_inspection', reason='Terminal outcome is not a proved unallocated 503; application failures must be inspected.')
            return
        for remaining in range(args.retry_delay, 0, -15):
            save('capacity_cooldown', retry_in_s=remaining)
            time.sleep(min(15, remaining))
        try:
            usage = subprocess.run(['colab', 'usage'], cwd=ROOT, capture_output=True, text=True, timeout=40, check=True).stdout
            active = re.search(r'Active assignments:\s*(\d+)', usage)
            if not active or int(active.group(1)) != 0:
                save('needs_inspection', reason='Fresh account observation does not prove zero assignments; no new request sent.')
                return
        except subprocess.SubprocessError:
            save('waiting_for_capacity', reason='Fresh assignment observation failed; retry observation without creating a runtime.')
            continue
        logfile = ROOT / '.colab-work' / f'capacity_queue_{len(report["restarts"])+1}.log'
        with logfile.open('a') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/run_remaining_queue.py',
                '--adopt', 'research/runs/qwen_zs_20261109T133608Z/receipt.json',
                '--stages', 'train_formal', 'formal_mil', 'openjev_zs', 'scale'], cwd=ROOT, stdout=stream, stderr=stream)
        pid, start = child.pid, identity(child.pid)
        assert start, 'New coordinator did not start'
        report['restarts'].append({'pid': pid, 'after_receipt': str(receipt_path.relative_to(ROOT)), 'log': str(logfile.relative_to(ROOT))})
        report.pop('reason', None)
        report.pop('retry_in_s', None)
        time.sleep(2)


if __name__ == '__main__':
    main()
