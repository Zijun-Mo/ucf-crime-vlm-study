"""Recover idempotent setup after a lost observation, without restarting a VM."""
import subprocess
import time


def execute_setup_with_evidence(execute, observe, recovery_seconds=120, clock=time.monotonic, sleep=time.sleep):
    try:
        execute()
        return 'execution_returned'
    except (RuntimeError, subprocess.TimeoutExpired):
        pass
    deadline = clock() + recovery_seconds
    retried = False
    last = {}
    while clock() < deadline:
        try:
            last = observe()
        except Exception as error:
            last = {'observation_error': type(error).__name__}
        if last.get('assignment_present') is False:
            raise RuntimeError('Actual setup assignment is missing; no retry was sent')
        if last.get('progress_http_status') == 200 and 'progress' in last:
            return 'verified_completion_marker'
        kernels = last.get('kernels', [])
        idle = last.get('kernels_http_status') == 200 and kernels and all(k.get('execution_state') == 'idle' for k in kernels)
        if idle and last.get('progress_http_status') == 404 and not retried:
            # The caller supplies an idempotent setup step only. Never submit
            # another cell while the actual kernel remains busy or unknown.
            retried = True
            try:
                execute()
                return 'idle_kernel_retry_returned'
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        sleep(5)
    raise TimeoutError('Setup observation recovery budget exhausted; actual execution is unproven: ' + str(last))
