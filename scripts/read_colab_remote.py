"""Read small remote diagnostics through fresh Colab assignments, without executing cells."""
import argparse
import base64
import json
from pathlib import Path
import sys
from urllib.parse import quote
import requests
from colab_cli.common import state
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from openjev_dss.cloud_io import sanitize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('session')
parser.add_argument('paths', nargs='+')
parser.add_argument('--endpoint')
parser.add_argument('--tail', type=int, default=8000)
args = parser.parse_args()
local = state.store.get(args.session)
endpoint = local.endpoint if local else args.endpoint
assignment = next((a for a in state.client.list_assignments() if a.endpoint == endpoint), None)
if assignment is None:
    raise SystemExit('Backend assignment missing; no file request sent')
proxy = assignment.runtime_proxy_info
parameters = {'authuser': '0', 'colab-runtime-proxy-token': proxy.token}
with requests.Session() as client:
    for path in args.paths:
        report = {'path': path}
        try:
            response = client.get(proxy.url.rstrip('/') + '/api/contents/' + quote(path.lstrip('/'), safe='/'),
                                  params={**parameters, 'content': '1'}, timeout=15)
            report['http_status'] = response.status_code
            if response.status_code == 200:
                payload = response.json()
                if payload['type'] == 'directory':
                    report['entries'] = [{k: row.get(k) for k in ['name', 'type', 'size']} for row in payload['content']]
                else:
                    text = payload['content']
                    if payload.get('format') == 'base64':
                        text = base64.b64decode(text).decode(errors='replace')
                    report['text'] = text[-args.tail:]
        except requests.RequestException as error:
            report['observation_error'] = type(error).__name__
        print(sanitize(json.dumps(report, ensure_ascii=False)), flush=True)
