"""Read the live server assignment, real Jupyter state and bounded progress JSON.

Run with the Python interpreter installed alongside the `colab` executable.
Unlike the CLI's display flag, execution_state here comes from /api/kernels.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import requests
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from openjev_dss.cloud_io import sanitize
from colab_cli.common import state

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('session')
parser.add_argument('--progress')
parser.add_argument('--worker', action='store_true')
parser.add_argument('--ssh', action='store_true', help='Opt into independent SSH fallback; only one SSH connection per runtime is allowed')
parser.add_argument('--endpoint', help='Recover observation if a transient CLI error pruned only local metadata')
args = parser.parse_args()
report = {'session': args.session}
try:
    assignments = state.client.list_assignments()
    session = state.store.get(args.session)
    endpoint = session.endpoint if session else args.endpoint
    if endpoint is None or endpoint not in {a.endpoint for a in assignments}:
        report['assignment_present'] = False
    else:
        report['assignment_present'] = True
        report['endpoint'] = endpoint
        assignment = next(a for a in assignments if a.endpoint == endpoint)
        proxy = assignment.runtime_proxy_info
        session = SimpleNamespace(url=proxy.url, token=proxy.token,
                                  kernel_id=session.kernel_id if session else None)
        parameters = {'authuser': '0', 'colab-runtime-proxy-token': session.token}
        response = requests.get(session.url.rstrip('/') + '/api/kernels', params=parameters, timeout=7)
        report['kernels_http_status'] = response.status_code
        if response.status_code == 200:
            report['kernels'] = [{k: row.get(k) for k in ['id', 'execution_state', 'last_activity', 'connections']}
                                 for row in response.json()]
            report['cached_kernel_id'] = session.kernel_id
        if args.progress:
            from urllib.parse import quote
            response = requests.get(session.url.rstrip('/') + '/api/contents/' + quote(args.progress.lstrip('/'), safe='/'),
                                    params={**parameters, 'content': '1'}, timeout=7)
            report['progress_http_status'] = response.status_code
            if response.status_code == 200:
                payload = response.json()
                text = payload['content']
                if payload.get('format') == 'base64':
                    text = base64.b64decode(text).decode()
                data = json.loads(text)
                keys = ['status', 'stage', 'job', 'group', 'completed', 'updated', 'steps', 'validation_videos',
                        'active_job', 'active_status', 'elapsed_s', 'weights', 'error', 'epoch_cursor', 'step_cursor', 'epoch_limit', 'test_used', 'validation_progress',
                        'teacher_videos','video_cursor','best_epoch','best_localization','baseline','validation','last_epoch_active_hours',
                        'completed_videos','total_videos','scored_windows','optimizer_updates','active_arm']
                report['progress'] = {k: data[k] for k in keys if k in data}
                if 'arms' in data:
                    report['progress']['arms'] = {name:{k:a[k] for k in ['steps','best_epoch','best_ap','stopped','stop_reason','bad_epochs'] if k in a}
                        | {'validation_epochs':len(a['validation'])} for name,a in data['arms'].items()}
        if args.worker:
            def fetch_text(path):
                from urllib.parse import quote
                r = requests.get(session.url.rstrip('/') + '/api/contents/' + quote(path.lstrip('/'), safe='/'),
                                 params={**parameters, 'content': '1'}, timeout=7)
                if r.status_code != 200:
                    return r.status_code, None
                payload = r.json()
                text = payload['content']
                if payload.get('format') == 'base64':
                    text = base64.b64decode(text).decode()
                return 200, text
            status, text = fetch_text('/content/ucf_background_handle.json')
            report['worker_handle_http_status'] = status
            if status == 200:
                handle = json.loads(text)
                report['worker_handle'] = handle
                status, text = fetch_text('/proc/' + str(handle['pid']) + '/stat')
                report['worker_proc_http_status'] = status
                if status == 200:
                    fields = text.split(') ', 1)[1].split()
                    report['worker_state'] = fields[0]
                    report['worker_start_matches'] = fields[19] == handle['start_ticks']
                    report['worker_alive'] = fields[0] != 'Z' and report['worker_start_matches']
                elif status == 404:
                    report['worker_alive'] = False
except Exception as error:
    report['observation_error'] = type(error).__name__
# SSH is an independent observation channel; require an existing local session
# so the CLI's convenience auto-create behavior cannot allocate a new VM here.
if (args.ssh and args.worker and report.get('assignment_present') is True and
        'worker_alive' not in report and state.store.get(args.session) is not None):
    import subprocess
    import shlex
    key = Path.home() / '.ssh/id_ed25519'
    if key.is_file():
        source = """import json
from pathlib import Path
r={}
p=Path(PROGRESS_PATH)
if p.is_file():
    r['progress']=json.loads(p.read_text())
p=Path('/content/ucf_background_handle.json')
if p.is_file():
    h=json.loads(p.read_text());r['worker_handle']=h
    p=Path('/proc')/str(h['pid'])/'stat'
    if p.is_file():
        f=p.read_text().split(') ',1)[1].split()
        r.update(worker_state=f[0],worker_start_matches=f[19]==h['start_ticks'])
        r['worker_alive']=f[0]!='Z' and r['worker_start_matches']
    else:r['worker_alive']=False
print(json.dumps(r))
""".replace('PROGRESS_PATH', repr(args.progress or '/content/ucf_group_status.json'))
        known = Path(__file__).resolve().parents[1] / '.colab-work/ssh_known_hosts'
        command = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile='+str(known),
                   '-o', 'HostKeyAlias='+report['endpoint'],
                   '-o', 'ProxyCommand=colab ssh --proxy-mode -s '+shlex.quote(args.session),
                   '-i', str(key), 'root@colab-runtime', 'python3', '-']
        try:
            remote = subprocess.run(command, input=source, capture_output=True, text=True, timeout=25, check=True)
            report.update(json.loads(remote.stdout))
            if 'arms' in report.get('progress', {}):
                p = report['progress']
                p.pop('feature_rows', None)
                for a in p['arms'].values():
                    a['validation_epochs'] = len(a.get('validation', []))
                    for key in ['rows', 'validation', 'epoch_artifacts']:a.pop(key, None)
            report['worker_observation_channel'] = 'colab_cli_ssh'
        except Exception as error:
            report['ssh_observation_error'] = type(error).__name__
print(sanitize(json.dumps(report, ensure_ascii=False, indent=2)))
