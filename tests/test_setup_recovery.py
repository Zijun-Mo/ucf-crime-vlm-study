import subprocess
import pytest
from openjev_dss.setup_recovery import execute_setup_with_evidence


def test_busy_kernel_is_observed_until_marker_without_duplicate_execution():
    calls=[]
    def execute():
        calls.append(1)
        raise subprocess.TimeoutExpired('mount',90)
    observations=iter([
        {'assignment_present':True,'kernels_http_status':200,'kernels':[{'execution_state':'busy'}],'progress_http_status':404},
        {'assignment_present':True,'progress_http_status':200,'progress':{}},
    ])
    assert execute_setup_with_evidence(execute,lambda:next(observations),sleep=lambda _:None)=='verified_completion_marker'
    assert len(calls)==1


def test_only_observed_idle_kernel_and_missing_marker_allow_one_idempotent_retry():
    calls=[]
    def execute():
        calls.append(1)
        if len(calls)==1:raise RuntimeError('Connection was lost')
    observation={'assignment_present':True,'kernels_http_status':200,'kernels':[{'execution_state':'idle'}],'progress_http_status':404}
    assert execute_setup_with_evidence(execute,lambda:observation)=='idle_kernel_retry_returned'
    assert len(calls)==2


def test_unknown_observation_never_resubmits_a_cell():
    calls=[];ticks=iter([0,0,1,2,4])
    def execute():calls.append(1);raise RuntimeError('Connection was lost')
    with pytest.raises(TimeoutError,match='unproven'):
        execute_setup_with_evidence(execute,lambda:{'observation_error':'ReadTimeout'},recovery_seconds=3,clock=lambda:next(ticks),sleep=lambda _:None)
    assert len(calls)==1


def test_actual_missing_assignment_is_terminal_without_resubmission():
    def execute():raise RuntimeError('Connection was lost')
    with pytest.raises(RuntimeError,match='Actual setup assignment is missing'):
        execute_setup_with_evidence(execute,lambda:{'assignment_present':False})
