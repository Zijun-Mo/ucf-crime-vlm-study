import sys
import pytest
from openjev_dss.cloud_io import checked, sanitize


def test_transfer_failure_keeps_reason_and_removes_signed_credentials(tmp_path):
    secret_url = 'https://runtime.example/file?colab-runtime-proxy-token=private123&x=4'
    script = "print('HTTP 403 ' + " + repr(secret_url) + ");raise SystemExit(1)"
    log = tmp_path / 'transfer.log'
    with pytest.raises(RuntimeError) as error:
        checked([sys.executable, '-c', script], log=log)
    assert '403' in str(error.value) and 'runtime.example/file' in str(error.value)
    assert 'private123' not in str(error.value) and 'private123' not in log.read_text()


def test_large_cache_can_upload_longer_than_old_three_minute_cap(tmp_path, monkeypatch):
    import subprocess
    from openjev_dss import cloud_io
    monkeypatch.delenv('UCF_DEADLINE', raising=False)
    cache=tmp_path/'long_video.pt'
    # Sparse test file: emulate the real 9B cache size without allocating its bytes.
    with cache.open('wb') as stream:stream.truncate(3601*49*4096*2)
    def upload(command, **kwargs):
        if kwargs['timeout'] < 240:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, stdout='uploaded')
    monkeypatch.setattr(cloud_io.subprocess, 'run', upload)
    assert cloud_io.checked(['rclone','copyto',str(cache),'gdrive:cache'],timeout=180,log=tmp_path/'log').stdout=='uploaded'


def test_transfer_reserves_time_for_durable_status_and_runtime_release(tmp_path, monkeypatch):
    import time
    from openjev_dss import cloud_io
    cache=tmp_path/'large.pt'
    with cache.open('wb') as stream:stream.truncate(512*1024**2)
    monkeypatch.setenv('UCF_DEADLINE',str(time.time()+200))
    def forbidden(*args,**kwargs):raise AssertionError('upload started without enough time')
    monkeypatch.setattr(cloud_io.subprocess,'run',forbidden)
    with pytest.raises(TimeoutError,match='budget reserve'):
        cloud_io.checked(['rclone','copyto',str(cache),'gdrive:cache'],timeout=180,log=tmp_path/'log')


def test_small_result_bundle_keeps_requested_timeout(tmp_path, monkeypatch):
    from openjev_dss.cloud_io import transfer_timeout
    monkeypatch.delenv('UCF_DEADLINE',raising=False)
    result=tmp_path/'result.zip';result.write_bytes(b'zip')
    assert transfer_timeout(['rclone','copyto',str(result),'gdrive:result'],60)==60


def test_small_result_can_still_be_saved_inside_final_three_minutes(tmp_path,monkeypatch):
    import time
    from openjev_dss.cloud_io import transfer_timeout
    result=tmp_path/'status.json';result.write_text('{}')
    monkeypatch.setenv('UCF_DEADLINE',str(time.time()+200))
    allowed=transfer_timeout(['rclone','copyto',str(result),'gdrive:status'],180)
    assert 100 < allowed < 111


def test_timeout_preserves_the_native_reason_without_signed_credentials(tmp_path,monkeypatch):
    import subprocess
    from openjev_dss import cloud_io
    monkeypatch.delenv('UCF_DEADLINE',raising=False)
    def fail(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],1,output=b'HTTP403 rateLimitExceeded https://runtime.example/?token=private123')
    monkeypatch.setattr(cloud_io.subprocess,'run',fail)
    with pytest.raises(subprocess.TimeoutExpired):cloud_io.checked(['native','transfer'],timeout=1,log=tmp_path/'log')
    text=(tmp_path/'log').read_text()
    assert '403' in text and 'rateLimitExceeded' in text and 'private123' not in text


def test_capacity_retry_never_restarts_an_allocated_worker_or_unknown_timeout():
    from openjev_dss.cloud_io import unallocated_service_failure
    receipt={'allocation_requested_utc':'now','stop_exit_code':0,
             'native_drive_healthcheck':{'passed_before_gpu_allocation':True}}
    assert unallocated_service_failure(receipt,'POST assign: Service Unavailable')
    assert not unallocated_service_failure({**receipt,'allocated_utc':'now'},'Service Unavailable')
    assert not unallocated_service_failure(receipt,'Timed out waiting for assignment; outcome unknown')
    assert not unallocated_service_failure({**receipt,'stop_exit_code':1},'Service Unavailable')
