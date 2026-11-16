"""Capacity automation must never replace an uncertain or allocated worker."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def poll(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/poll_colab_capacity.py'
    spec = importlib.util.spec_from_file_location('capacity_poll_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / '.colab-work').mkdir()
    (tmp_path / 'research').mkdir()
    monkeypatch.setattr(sys, 'argv', ['poll', '--coordinator', '123', '--retry-delay', '120'])
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('A duplicate coordinator/runtime was requested')
    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    return module


def fixture(poll, **updates):
    receipt = {'controller_pid': 456, 'session': 'fixture', 'finished_utc': 'now',
               'allocation_requested_utc': 'now', 'stop_exit_code': 0,
               'native_drive_healthcheck': {'passed_before_gpu_allocation': True}, **updates}
    (poll.ROOT / 'research/receipt.json').write_text(json.dumps(receipt))
    (poll.ROOT / 'research/controller.log').write_text('Service Unavailable')
    (poll.ROOT / 'research/queue_state.json').write_text(json.dumps({'adopted_receipt': 'research/receipt.json'}))


def terminal_coordinator(poll, monkeypatch, controller_alive=False):
    calls = 0
    def identity(pid):
        nonlocal calls
        if pid == 123:
            calls += 1
            return 'start' if calls == 1 else None
        return 'worker-start' if controller_alive else None
    monkeypatch.setattr(poll, 'identity', identity)


def test_missing_coordinator_cannot_replace_a_live_allocation_controller(poll, monkeypatch):
    fixture(poll, finished_utc=None)
    terminal_coordinator(poll, monkeypatch, controller_alive=True)
    poll.main()
    state = json.loads((poll.ROOT / 'research/capacity_poll_state.json').read_text())
    assert state['status'] == 'needs_inspection' and not state['restarts']


def test_fresh_nonzero_assignment_prevents_retry_even_after_real_503(poll, monkeypatch):
    fixture(poll)
    terminal_coordinator(poll, monkeypatch)
    monkeypatch.setattr(poll.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='Active assignments: 1'))
    poll.main()
    state = json.loads((poll.ROOT / 'research/capacity_poll_state.json').read_text())
    assert state['status'] == 'needs_inspection' and 'zero assignments' in state['reason']


def test_actual_kernel_connection_finishes_polling_without_new_allocation(poll, monkeypatch):
    fixture(poll, allocated_utc='now', finished_utc=None)
    monkeypatch.setattr(poll, 'identity', lambda _: 'start')
    observed = {'assignment_present': True, 'kernels_http_status': 200, 'kernels': [{'id': 'real-kernel'}]}
    monkeypatch.setattr(poll.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(observed)))
    poll.main()
    state = json.loads((poll.ROOT / 'research/capacity_poll_state.json').read_text())
    assert state['status'] == 'connected' and not state['restarts']
