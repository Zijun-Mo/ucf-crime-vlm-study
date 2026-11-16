"""Initialize/adopt one idle kernel on an already allocated, identified VM.

Never allocate a runtime or submit a cell. An ambiguous kernel-creation response
is resolved by reading the kernel list; the POST is never repeated.
"""
import argparse,json,time,uuid
from urllib.parse import urlencode
import requests
from colab_cli.common import state


def wake_kernel(proxy,kernel_id):
    """A metadata-only request makes a freshly created kernel publish idle."""
    import websocket
    session=uuid.uuid4().hex;message_id=uuid.uuid4().hex
    url=proxy.url.rstrip('/').replace('https://','wss://')+'/api/kernels/'+kernel_id+'/channels?'+urlencode(
        {'authuser':'0','colab-runtime-proxy-token':proxy.token,'session_id':session})
    ws=websocket.create_connection(url,timeout=10,header=['X-Colab-Client-Agent: colab-cli','X-Colab-Runtime-Proxy-Token: '+proxy.token],origin='https://colab.research.google.com')
    try:
        ws.send(json.dumps(dict(header=dict(msg_id=message_id,username='colab-cli',session=session,msg_type='kernel_info_request',version='5.3'),parent_header={},metadata={},content={},channel='shell',buffers=[])))
        for _ in range(8):
            msg=json.loads(ws.recv())
            if msg.get('parent_header',{}).get('msg_id')==message_id:return
    finally:ws.close(timeout=1)


def initialize(session,endpoint,timeout=60):
    local=state.store.get(session)
    assert local and local.endpoint==endpoint,'Known allocated session required'
    assignment=next((a for a in state.client.list_assignments() if a.endpoint==endpoint),None)
    assert assignment is not None,'Original assignment missing; never allocate'
    proxy=assignment.runtime_proxy_info
    params={'authuser':'0','colab-runtime-proxy-token':proxy.token}
    # Match the installed Colab CLI's Jupyter manager. GET accepts the proxy
    # query token, while kernel creation also requires its authenticated headers.
    headers={'Authorization':'Bearer '+proxy.token,'X-Colab-Client-Agent':'colab-cli',
        'X-Colab-Runtime-Proxy-Token':proxy.token,'Accept':'application/json','Content-Type':'application/json'}
    url=proxy.url.rstrip('/')+'/api/kernels'
    def kernels():
        for attempt in range(3):
            try:
                r=requests.get(url,params=params,headers=headers,timeout=7);r.raise_for_status();return r.json()
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(1)
    rows=kernels();created=False
    if not rows:
        created=True
        try:
            r=requests.post(url,params=params,headers=headers,json={'name':'python3','path':None},timeout=15)
            r.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError('Kernel creation rejected with HTTP '+str(exc.response.status_code)) from None
        except requests.RequestException:
            # Delivery might have succeeded. Observe; do not send another POST.
            pass
    deadline=time.monotonic()+timeout;probes=0
    while True:
        try:rows=kernels()
        except requests.RequestException:
            if time.monotonic()>deadline:raise
            time.sleep(1);continue
        assert len(rows)<=1,'Multiple kernels: inspect instead of guessing'
        if rows and rows[0]['execution_state']=='idle':
            local.kernel_id=rows[0]['id'];local.url=proxy.url;local.token=proxy.token
            state.store.add(local)
            return dict(kernel_id=local.kernel_id,idle=True,created=created,new_vm_allocations=0,submitted_cells=0,metadata_probes=probes)
        assert not rows or rows[0]['execution_state']=='starting','Kernel is busy; do not duplicate execution'
        if rows and probes<3:
            probes+=1
            try:wake_kernel(proxy,rows[0]['id'])
            except Exception:pass # Read-only metadata may be retried; never an execute_request.
        if time.monotonic()>deadline:raise TimeoutError('Kernel initialization unresolved; observed '+str([{k:r.get(k) for k in ['id','execution_state']} for r in rows])+'; inspect same VM')
        time.sleep(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('endpoint');a=p.parse_args()
    print(json.dumps(initialize(a.session,a.endpoint)))
