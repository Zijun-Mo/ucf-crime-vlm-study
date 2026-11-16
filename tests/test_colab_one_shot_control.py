"""Transport failures must never duplicate a remotely submitted control cell."""
import importlib.util,sys,types
from pathlib import Path
import pytest


def module(monkeypatch,*,busy=False,complete=True):
    proxy=types.SimpleNamespace(url='https://fixture.invalid',token='fixture')
    local=types.SimpleNamespace(endpoint='existing',kernel_id='kernel')
    state=types.SimpleNamespace(store=types.SimpleNamespace(get=lambda name:local),
        client=types.SimpleNamespace(list_assignments=lambda:[types.SimpleNamespace(endpoint='existing',runtime_proxy_info=proxy)]))
    monkeypatch.setitem(sys.modules,'colab_cli.common',types.SimpleNamespace(state=state))
    sent=[]
    class Socket:
        def send(self,s):sent.append(s);raise ConnectionResetError('acknowledgement lost after delivery')
        def close(self,**kw):pass
    monkeypatch.setitem(sys.modules,'websocket',types.SimpleNamespace(create_connection=lambda *a,**kw:Socket()))
    name='one_shot_fixture';path=Path(__file__).resolve().parents[1]/'scripts/execute_colab_control.py'
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Reply:
        status_code=200
        def __init__(self,data):self.data=data
        def raise_for_status(self):pass
        def json(self):return self.data
    def get(url,**kw):
        if url.endswith('/api/kernels'):return Reply([dict(id='kernel',execution_state='busy' if busy else 'idle')])
        if url.endswith('.done') and complete:return Reply(dict(content='complete',format='text'))
        r=Reply({});r.status_code=404;return r
    monkeypatch.setattr(m.requests,'get',get)
    clock=[0.]
    monkeypatch.setattr(m.time,'time',lambda:clock[0])
    monkeypatch.setattr(m.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    return m,sent


def test_ambiguous_ws_ack_uses_durable_marker_without_resending(monkeypatch):
    m,sent=module(monkeypatch)
    assert m.execute('fixture','existing',"print('fixture')",'fixture.py',10)['complete']
    assert len(sent)==1


def test_unresolved_delivery_times_out_without_duplicate_submission(monkeypatch):
    m,sent=module(monkeypatch,complete=False)
    with pytest.raises(TimeoutError,match='never automatically resubmit'):
        m.execute('fixture','existing','pass','fixture.py',8)
    assert len(sent)==1


def test_busy_kernel_does_not_receive_another_control_cell(monkeypatch):
    m,sent=module(monkeypatch,busy=True)
    with pytest.raises(AssertionError,match='idle'):m.execute('fixture','existing','pass','fixture.py',8)
    assert not sent
