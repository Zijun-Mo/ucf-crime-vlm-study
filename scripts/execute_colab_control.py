"""Submit one marked control cell to an EXISTING kernel, observe via fresh HTTP.

Use Colab CLI's own credential/state client. A websocket acknowledgement loss
does not trigger resubmission: inspect the durable completion/error marker and
kernel state first. Never allocate a VM, create a kernel, or start a trainer here.
Run with the Python interpreter bundled with the installed Colab CLI.
"""
import argparse,base64,json,sys,time,uuid
from pathlib import Path
from urllib.parse import urlencode,quote
import requests,websocket
from colab_cli.common import state
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize


def execute(session,endpoint,source,filename,timeout=300):
    local=state.store.get(session)
    assert local and local.endpoint==endpoint,'Existing CLI session required; no allocation'
    token=uuid.uuid4().hex;done='/content/ucf_control_'+token+'.done'
    error='/content/ucf_control_'+token+'.error';started='/content/ucf_control_'+token+'.started'
    code=("import pathlib as _v4_path,traceback as _v4_trace\n"
        f"if not _v4_path.Path({done!r}).exists():\n"
        f"    _v4_path.Path({started!r}).write_text('started')\n"
        "    try:\n"
        f"        exec(compile({source!r},{str(filename)!r},'exec'),globals(),globals())\n"
        f"        _v4_path.Path({done!r}).write_text('complete')\n"
        "    except BaseException:\n"
        f"        _v4_path.Path({error!r}).write_text(_v4_trace.format_exc())\n"
        "        raise\n")
    def proxy():
        a=next((a for a in state.client.list_assignments() if a.endpoint==endpoint),None)
        assert a is not None,'Original assignment missing; never recreate'
        return a.runtime_proxy_info
    def read(path,p):
        r=requests.get(p.url.rstrip('/')+'/api/contents/'+quote(path.lstrip('/'),safe='/'),
            params={'authuser':'0','colab-runtime-proxy-token':p.token,'content':'1'},timeout=10)
        if r.status_code==404:return None
        r.raise_for_status();d=r.json();s=d['content']
        return base64.b64decode(s).decode() if d.get('format')=='base64' else s
    # These reads precede any cell submission, so transport retries are safe.
    for attempt in range(3):
        try:
            p=proxy();params={'authuser':'0','colab-runtime-proxy-token':p.token}
            r=requests.get(p.url.rstrip('/')+'/api/kernels',params=params,timeout=10);r.raise_for_status()
            break
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2)
    kernels=r.json();kernel=next((k for k in kernels if k['id']==local.kernel_id),None)
    assert kernel and kernel['execution_state']=='idle','Known kernel must be idle; no duplicate execution'
    sent=False;submitted=time.time();last_error=None
    while time.time()-submitted<timeout:
        if not sent:
            ws=None
            try:
                p=proxy();url=p.url.rstrip('/').replace('https://','wss://')+'/api/kernels/'+local.kernel_id+'/channels?'+urlencode({**params,'colab-runtime-proxy-token':p.token,'session_id':uuid.uuid4().hex})
                ws=websocket.create_connection(url,timeout=12,
                    header=['X-Colab-Client-Agent: colab-cli','X-Colab-Runtime-Proxy-Token: '+p.token],origin='https://colab.research.google.com')
                ident=uuid.uuid4().hex
                message=dict(header=dict(msg_id=ident,username='colab-cli',session=uuid.uuid4().hex,msg_type='execute_request',version='5.3'),
                    parent_header={},metadata={},content=dict(code=code,silent=False,store_history=False,user_expressions={},allow_stdin=False,stop_on_error=True),channel='shell',buffers=[])
                sent=True # From this point delivery is ambiguous, never blind retry.
                ws.send(json.dumps(message))
                # Give the proxy evidence that the kernel accepted this frame
                # before closing. An immediate TLS close can race forwarding.
                # Completion still comes from durable HTTP markers, never an
                # assumption that a websocket acknowledgement was delivered.
                try:
                    ws.settimeout(3)
                    for _ in range(6):
                        received=ws.recv()
                        if not received:break
                        msg=json.loads(received)
                        if msg.get('parent_header',{}).get('msg_id')==ident:break
                except Exception:pass
            except Exception as exc:last_error=sanitize(str(exc))[:1000]
            finally:
                if ws:
                    try:ws.close(timeout=1)
                    except Exception:pass
        try:
            p=proxy()
            if read(done,p)=='complete':return dict(complete=True,marker=done,seconds=time.time()-submitted,transport='CLI SDK state + single WS submission + fresh HTTP markers')
            failure=read(error,p)
            if failure is not None:raise RuntimeError(sanitize(failure))
        except requests.RequestException as exc:last_error=type(exc).__name__
        time.sleep(3)
    raise TimeoutError(f'Control delivery/acknowledgement unresolved; inspect {done}, {started}, {error}; last={last_error}; never automatically resubmit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('endpoint');p.add_argument('file',type=Path);p.add_argument('--timeout',type=int,default=300);a=p.parse_args()
    try:print(json.dumps(execute(a.session,a.endpoint,a.file.read_text(),a.file,a.timeout)),flush=True)
    except Exception:
        import traceback
        print(sanitize(traceback.format_exc()),file=sys.stderr,flush=True)
        raise SystemExit(1)
