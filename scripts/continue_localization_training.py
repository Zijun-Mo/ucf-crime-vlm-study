"""Supervise one trajectory across bounded Colab allocations, without sweeps."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.learning import atomic_json
from openjev_dss.cloud_io import unallocated_service_failure


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-receipt',type=Path,required=True)
    parser.add_argument('--allocations',type=int,default=10)
    args=parser.parse_args()
    lock=(ROOT/'.colab-work/localization_training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt_path=args.current_receipt.resolve()
    initial_receipt=json.loads(receipt_path.read_text())
    policy_path=initial_receipt.get('localization_policy','research/localization_v1/policy.json')
    continuation_path=initial_receipt.get('continuation_control')
    state_path=(ROOT/(continuation_path or policy_path)).parent/'controller_state.json'
    state=dict(status='running',supervisor_pid=os.getpid(),current_receipt=str(receipt_path.relative_to(ROOT)),
        trajectory='one fixed Qwen localization protocol; optimizer resumed, never reinitialized after committed updates',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),continuations=[])
    failures=0
    for number in range(args.allocations):
        receipt=json.loads(receipt_path.read_text())
        while 'finished_utc' not in receipt:
            if not alive(receipt['controller_pid']):
                state.update(status='needs_inspection',reason='Controller exited without releasing/settling its assignment')
                atomic_json(state_path,state);return 1
            state['current_receipt']=str(receipt_path.relative_to(ROOT));atomic_json(state_path,state)
            time.sleep(20);receipt=json.loads(receipt_path.read_text())
        result_path=receipt_path.parent/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        if result.get('status') in ['complete','budget_complete']:
            state.update(status=result['status'],training_steps=result.get('steps'),best_epoch=result.get('best_epoch'),
                selection_status=result.get('selection_status'),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic_json(state_path,state);return 0
        resume=result.get('status')=='budget_stopped'
        if result.get('status')=='prepared':
            # Reviewed annotations are now archived in every new payload. A
            # transient FUSE directory-cache miss can therefore be resumed once.
            policy=json.loads((ROOT/policy_path).read_text())
            resume=(ROOT/policy['validation']['annotations']).exists()
        if not resume:
            log=(receipt_path.parent/'controller.log').read_text(errors='replace')
            retry=unallocated_service_failure(receipt,log)
            failures+=1
            if not retry or failures>3:
                state.update(status='needs_inspection',reason=receipt.get('error',result.get('status','Unknown failure')))
                atomic_json(state_path,state);return 1
            time.sleep(60)
        else:failures=0
        if continuation_path:
            from openjev_dss.followup_budget import available
            import re
            rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt.get('usage_allocated',''))
            hours=result.get('last_epoch_active_hours',2.)
            required=(float(hours)*1.3+.35)*float(rate.group(1))+.2 if rate else float('inf')
            remaining=available(ROOT,'v3_training')
            if remaining<required:
                state.update(status='budget_exhausted',reason='Remaining authorized CU cannot fund another conservative complete epoch with setup and release margin',remaining_training_cu=remaining,required_next_allocation_cu=required,converged=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                atomic_json(state_path,state);return 0
        before=set((ROOT/'research/runs').glob('localization_train_*/receipt.json'))
        state['continuations'].append(dict(previous_receipt=str(receipt_path.relative_to(ROOT)),
            previous_status=result.get('status'),next_phase='auto',optimizer_resume=True))
        atomic_json(state_path,state)
        command=[sys.executable,str(ROOT/'scripts/launch_study_job.py'),'localization_train','--gpu','L4',
                 '--minutes','720','--localization-phase','auto','--localization-policy',policy_path]
        if continuation_path:command+=['--continuation',continuation_path]
        subprocess.run(command,cwd=ROOT,check=False)
        new=set((ROOT/'research/runs').glob('localization_train_*/receipt.json'))-before
        if len(new)!=1:
            state.update(status='needs_inspection',reason='Expected exactly one continuation receipt')
            atomic_json(state_path,state);return 1
        receipt_path=new.pop()
    state.update(status='allocation_limit_reached',reason='No more allocations launched; inspect durable state')
    atomic_json(state_path,state);return 1


if __name__=='__main__':raise SystemExit(main())
