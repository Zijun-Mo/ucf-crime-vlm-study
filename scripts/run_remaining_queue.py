"""Serially finish verified study groups; adopt an existing controller without duplication."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adopt', type=Path, required=True)
parser.add_argument('--stages', nargs='+', default=['train_formal', 'formal_mil', 'openjev_zs', 'scale'])
parser.add_argument('--defer-final-analysis', action='store_true', help='Keep GPU inference running while formal training protocol is revised')
args = parser.parse_args()
assert set(args.stages) <= {'qwen_zs', 'openjev_zs', 'train', 'qwen_mil', 'scale', 'train_formal', 'formal_mil'}
lock = (ROOT / '.colab-work/remaining_queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = ROOT / 'research/queue_state.json'
state = {'status': 'running', 'pid': os.getpid(), 'started': time.time(), 'completed': [],
         'coordinator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'stages': args.stages, 'adopted_receipt': str(args.adopt),
         'scope': 'matched zero-shot and scale inference plus both validation-converged formal MIL arms; historical one-epoch MIL remains auxiliary; optional 27B follows a separate resource audit'}


def save():
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(state_path)


def read_receipt(path):
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.1)


def pid_alive(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0] != 'Z'
    except FileNotFoundError:
        return False


def assert_released(receipt):
    # An allocation can be rejected before it has an endpoint. In that case
    # require a fresh account-wide zero-assignment observation.
    if not receipt.get('endpoint'):
        output = subprocess.run(['colab', 'usage'], capture_output=True, text=True, timeout=60, check=True).stdout
        count = re.search(r'Active assignments:\s*(\d+)', output)
        if not count or int(count.group(1)) != 0:
            raise RuntimeError('Failed allocation release is unverified; do not allocate another GPU')
        return
    # Fresh server observation, not CLI's cached IDLE/RUNNING display.
    command = ['/opt/colab-cli/bin/python',
               'scripts/inspect_colab_job.py', receipt['session'], '--endpoint', receipt['endpoint']]
    report = json.loads(subprocess.run(command, capture_output=True, text=True, timeout=70, check=True).stdout)
    if report.get('assignment_present') is not False:
        raise RuntimeError('Previous runtime release is unverified; do not allocate another GPU')


def wait_for_receipt(path):
    receipt = read_receipt(path)
    pid = receipt['controller_pid']
    while pid_alive(pid):
        state.update(active_receipt=str(path), active_job=receipt['job'])
        save()
        time.sleep(20)
        receipt = read_receipt(path)
    receipt = read_receipt(path)
    if 'finished_utc' not in receipt:
        raise RuntimeError('Controller exited without terminal receipt; inspect actual remote worker before resuming')
    assert_released(receipt)
    receipt['receipt_path'] = str(path.relative_to(ROOT) if path.is_absolute() else path)
    return receipt


def run_stage(stage, minutes=360, gpu='L4', probe=False, epoch_limit=None):
    sys.path.insert(0,str(ROOT/'src'))
    for allocation_attempt in range(3):
        prefix = stage + ('_probe' if probe else '')
        before = set((ROOT / 'research/runs').glob(prefix + '_*/receipt.json'))
        with (ROOT / '.colab-work' / ('queue_' + prefix + '_' + gpu + '.log')).open('a') as log:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/launch_study_job.py',
                                      stage, '--gpu', gpu, '--minutes', str(minutes)] + (['--probe'] if probe else []) +
                                      (['--epoch-limit', str(epoch_limit)] if epoch_limit is not None else []), stdout=log, stderr=log)
            while child.poll() is None:
                candidates = set((ROOT / 'research/runs').glob(prefix + '_*/receipt.json')) - before
                if candidates:
                    assert len(candidates) == 1
                    state.update(active_receipt=str(next(iter(candidates)).relative_to(ROOT)), active_job=stage)
                    save()
                time.sleep(20)
        candidates = set((ROOT / 'research/runs').glob(prefix + '_*/receipt.json')) - before
        if len(candidates) != 1:
            raise RuntimeError('Missing or ambiguous new receipt for ' + stage)
        receipt=wait_for_receipt(next(iter(candidates)))
        from openjev_dss.cloud_io import unallocated_service_failure
        log_text=(ROOT/receipt['receipt_path']).parent.joinpath('controller.log').read_text(errors='replace')
        if allocation_attempt<2 and unallocated_service_failure(receipt,log_text):
            state.setdefault('allocation_retries',[]).append({'receipt':receipt['receipt_path'],'reason':'Server503 before allocation; fresh zero-assignment check passed'})
            for remaining in range(120,0,-20):
                state.update(allocation_retry_wait_s=remaining,active_job=stage);save();time.sleep(20)
            state.pop('allocation_retry_wait_s',None)
            continue
        return receipt

def complete_or_resume(receipt):
    if receipt['status'] == 'complete':
        return receipt
    result_path = ROOT / state['active_receipt']
    result_path = result_path.parent / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    text = str(receipt.get('error', '')) + str(result.get('error', ''))
    progressed = bool(result.get('completed')) or result.get('active_status', {}).get('videos', 0) > 0 or result.get('steps', 0) > 0
    # One bounded retry only at a deliberate time boundary, preserving durable work.
    boundary = 'TimeoutError' in text or result.get('status') == 'budget_stopped'
    if not (progressed and boundary):
        raise RuntimeError('Actual failure requires inspection: ' + str(receipt.get('error', receipt['status'])))
    state['resuming'] = receipt['job']
    save()
    resumed = run_stage(receipt['job'], 360, receipt['gpu'])
    if resumed['status'] != 'complete':
        raise RuntimeError('Bounded resume did not finish; preserve partial evidence and inspect')
    return resumed



def formal_stage(initial_receipt=None):
    """Resume only demonstrated progress; extend only under the frozen val rule."""
    policy=json.loads((ROOT/'research/training_protocol_v2.json').read_text())
    limit=initial_receipt.get('epoch_limit',policy['epoch_budget']['initial']) if initial_receipt else policy['epoch_budget']['initial']
    previous_progress=None
    decisions=[]
    while True:
        usage=subprocess.run(['colab','usage'],capture_output=True,text=True,timeout=60,check=True).stdout
        match=re.search(r'Current balance:\s*([0-9.]+)',usage)
        assert match,'Missing actual balance; do not assume affordable allocation'
        balance=float(match.group(1))
        rates=[]
        for path in sorted((ROOT/'research/runs').glob('*/receipt.json')):
            r=read_receipt(path)
            m=re.search(r'Usage rate:\s*([0-9.]+)/hr',r.get('usage_allocated',''))
            if r.get('gpu')=='L4' and m and float(m.group(1))>0:rates.append(float(m.group(1)))
        assert rates,'No observed L4 rate'
        # Reserve two CU rather than knowingly exhausting the account; each
        # allocation has an independent hard watchdog and durable checkpoints.
        minutes=min(360,int((balance-2)/rates[-1]*60))
        if minutes<30:raise RuntimeError('Formal training requires more credits; durable state preserved')
        if initial_receipt is not None:
            receipt=initial_receipt
            initial_receipt=None
        else:
            receipt=run_stage('train_formal',minutes,'L4',epoch_limit=limit)
        path=ROOT/receipt['receipt_path'];result_path=path.parent/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        if receipt['status']=='complete':return receipt
        if result.get('status') not in ['budget_stopped','needs_extension']:
            raise RuntimeError('Formal training failure needs inspection: '+str(receipt.get('error')))
        assert result.get('generation',-1)>=0 and result.get('test_used') is False
        progress=(sum(a['steps'] for a in result['arms'].values()),
                  sum(len(a['validation']) for a in result['arms'].values()),
                  sum(result.get('validation_progress',{}).values()))
        if previous_progress is not None and progress<=previous_progress:
            raise RuntimeError('Formal training did not advance durable optimizer/validation work; inspect before another allocation')
        previous_progress=progress
        decision={'receipt':receipt['receipt_path'],'status':result['status'],'balance_before_cu':balance,
                  'epoch_limit':limit,'progress':list(progress),'test_used':False}
        if result['status']=='needs_extension':
            curves={name:[v['video_max_ap'] for v in arm['validation']] for name,arm in result['arms'].items()}
            assert max(map(len,curves.values()))==limit
            for arm in result['arms'].values():
                if not arm['stopped']:assert arm['bad_epochs']<policy['selection']['patience']
            limit+=policy['epoch_budget']['extension_block']
            decision.update(next_epoch_limit=limit,validation_curves=curves,
                            reason='Validation improvements still prevent the unchanged plateau rule; extend the epoch budget only')
        else:decision['reason']='Resume the same numerical contract from the durable allocation boundary'
        decisions.append(decision)
        (ROOT/'research/formal_training_continuations.json').write_text(json.dumps(decisions,indent=2)+'\n')
        state.update(formal_continuations='research/formal_training_continuations.json',formal_epoch_limit=limit)
        save()


def scale_stage():
    sys.path.insert(0, str(ROOT / 'src'))
    from openjev_dss.resource_planning import choose_scale_hardware
    model_config = json.loads((ROOT / 'research/model_resolution.json').read_text())['Qwen/Qwen3.5-9B']['config']
    upload = json.loads((ROOT / 'research/upload_benchmark.json').read_text())
    candidates = []
    decision = None
    for gpu in ['L4', 'G4', 'A100', 'H100']:
        availability=ROOT/'research/allocation_availability_observation.json'
        if availability.exists() and gpu in json.loads(availability.read_text())['account'].get('ineligibleGpus',[]):
            continue
        if gpu == 'H100' and decision is not None:
            break
        receipt = run_stage('qwen_9b', 15, gpu, probe=True)
        path = ROOT / receipt['receipt_path']
        result_path = path.parent / 'result.json'
        probe = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'unavailable'}
        candidates.append((probe, receipt))
        if probe.get('status') == 'error' and not any(p.get('status') == 'passed' for p, _ in candidates):
            error = probe.get('error', '')
            if not any(word in error for word in ['OutOfMemoryError', 'out of memory', 'not enough memory']):
                raise RuntimeError('Scale probe has an application failure; inspect before spending on another GPU: ' + error)
        try:
            decision = choose_scale_hardware(candidates, model_config, upload)
        except ValueError:
            decision = None
        state['scale_probe_receipts'] = [r['receipt_path'] for _, r in candidates]
        save()
    if decision is None:
        raise RuntimeError('No available GPU passed the unchanged 9B BF16/batch-4 resource gate')
    decision['probe_receipts'] = state['scale_probe_receipts']
    (ROOT / 'research/scale_9b_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    state['scale_decision'] = 'research/scale_9b_decision.json'
    save()
    return complete_or_resume(run_stage('qwen_9b', 360, decision['selected_gpu']))


try:
    save()
    adopted = wait_for_receipt(args.adopt)
    adopted = formal_stage(adopted) if adopted['job']=='train_formal' else complete_or_resume(adopted)
    state['completed'].append(adopted['job'])
    for stage in args.stages:
        if stage in state['completed']:
            continue
        state.update(active_job=stage)
        save()
        receipt = scale_stage() if stage == 'scale' else formal_stage() if stage == 'train_formal' else complete_or_resume(run_stage(stage))
        state['completed'].append(stage)
        if receipt['job'] not in state['completed']:
            state['completed'].append(receipt['job'])
        save()
    if args.defer_final_analysis:
        state.update(status='selected_inference_stages_complete', next='Formal multi-epoch MIL training and final analysis remain required')
    else:
        state.update(active_job='cpu_analysis')
        save()
        with (ROOT / '.colab-work/queue_analysis.log').open('a') as log:
            for script, flags in [('scripts/summarize_formal_training.py', []),
                                  ('scripts/recover_controlled_scores.py', ['--formal-v2', '--available-only']),
                                  ('scripts/analyze_controlled_study.py', ['--formal-v2', '--require-all']),
                                  ('scripts/plot_controlled_study.py', ['--formal-v2']),
                                  ('scripts/audit_compute_usage.py', [])]:
                subprocess.run([str(ROOT / '.venv/bin/python'), script, *flags], stdout=log, stderr=log, check=True)
        state.update(status='complete_mandatory_execution_and_analysis', next='Audit optional 27B resources and inspect actual figures, cost accounting, training receipts and overall completion')
except Exception as error:
    state.update(status='needs_inspection', error=type(error).__name__ + ': ' + str(error))
    raise
finally:
    save()
