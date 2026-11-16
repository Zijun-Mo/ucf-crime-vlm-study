import importlib.util,sys,types
from pathlib import Path
import pytest


def fixture(monkeypatch,initial,*,ambiguous=False):
    rows=list(initial);posts=[];stored=[]
    local=types.SimpleNamespace(endpoint='vm',kernel_id=None)
    proxy=types.SimpleNamespace(url='https://fixture.invalid',token='fixture')
    state=types.SimpleNamespace(store=types.SimpleNamespace(get=lambda _:local,add=stored.append),
        client=types.SimpleNamespace(list_assignments=lambda:[types.SimpleNamespace(endpoint='vm',runtime_proxy_info=proxy)]))
    monkeypatch.setitem(sys.modules,'colab_cli.common',types.SimpleNamespace(state=state))
    path=Path(__file__).resolve().parents[1]/'scripts/initialize_colab_kernel.py'
    spec=importlib.util.spec_from_file_location('kernel_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Reply:
        def raise_for_status(self):pass
        def json(self):return rows.copy()
    monkeypatch.setattr(m.requests,'get',lambda *a,**k:Reply())
    def post(*a,**k):
        posts.append(k);rows.append(dict(id='new',execution_state='idle'))
        if ambiguous:raise m.requests.Timeout('Reply lost after kernel created')
        return Reply()
    monkeypatch.setattr(m.requests,'post',post)
    return m,posts,stored


@pytest.mark.parametrize('ambiguous',[False,True])
def test_empty_fresh_vm_creates_once_even_if_acknowledgement_is_lost(monkeypatch,ambiguous):
    m,posts,stored=fixture(monkeypatch,[],ambiguous=ambiguous)
    r=m.initialize('session','vm')
    assert len(posts)==1 and len(stored)==1
    assert posts[0]['headers']['Authorization']=='Bearer fixture'
    assert posts[0]['headers']['X-Colab-Runtime-Proxy-Token']=='fixture'
    assert posts[0]['json']=={'name':'python3','path':None}
    assert r['kernel_id']=='new' and r['submitted_cells']==r['new_vm_allocations']==0


def test_adopts_single_idle_kernel_and_never_creates_alongside_busy_one(monkeypatch):
    m,posts,_=fixture(monkeypatch,[dict(id='old',execution_state='idle')])
    assert m.initialize('session','vm')['kernel_id']=='old' and not posts
    m,posts,_=fixture(monkeypatch,[dict(id='old',execution_state='busy')])
    with pytest.raises(AssertionError,match='busy'):m.initialize('session','vm')
    assert not posts


def test_transient_read_failure_after_creation_does_not_create_again(monkeypatch):
    m,posts,stored=fixture(monkeypatch,[])
    get=m.requests.get;reads=[0]
    def flaky(*args,**kwargs):
        reads[0]+=1
        if reads[0]==2:raise m.requests.exceptions.SSLError('Transient proxy EOF')
        return get(*args,**kwargs)
    monkeypatch.setattr(m.requests,'get',flaky)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    assert m.initialize('session','vm')['idle']
    assert len(posts)==len(stored)==1


def test_fresh_starting_kernel_uses_metadata_handshake_without_executing_cell(monkeypatch):
    m,posts,stored=fixture(monkeypatch,[dict(id='fresh',execution_state='starting')])
    get=m.requests.get;probes=[]
    def warm(proxy,kid):probes.append(kid)
    monkeypatch.setattr(m,'wake_kernel',warm)
    def ready(*args,**kwargs):
        r=get(*args,**kwargs)
        if probes:r.json=lambda:[dict(id='fresh',execution_state='idle')]
        return r
    monkeypatch.setattr(m.requests,'get',ready)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    result=m.initialize('session','vm')
    assert probes==['fresh'] and not posts and len(stored)==1
    assert result['metadata_probes']==1 and result['submitted_cells']==0
