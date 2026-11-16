"""Upload <=4 MiB through fresh CLI assignments without printing payloads/tokens."""
import argparse,base64,time
from pathlib import Path
from urllib.parse import quote
import requests
from colab_cli.common import state
p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('local',type=Path);p.add_argument('remote');a=p.parse_args()
payload=a.local.read_bytes();assert len(payload)<=4*1024**2
endpoint=state.store.get(a.session).endpoint
for attempt in range(3):
 try:
  assignment=next(x for x in state.client.list_assignments() if x.endpoint==endpoint)
  proxy=assignment.runtime_proxy_info;params={'authuser':'0','colab-runtime-proxy-token':proxy.token}
  url=proxy.url.rstrip('/')+'/api/contents/'+quote(a.remote.lstrip('/'),safe='/')
  r=requests.put(url,params=params,json={'type':'file','format':'base64','content':base64.b64encode(payload).decode()},timeout=25);r.raise_for_status()
  r=requests.get(url,params={**params,'content':'1','format':'base64'},timeout=25);r.raise_for_status()
  d=r.json();observed=base64.b64decode(d['content']) if d.get('format')=='base64' else d['content'].encode()
  assert observed==payload,'Upload verification failed'
  print('VERIFIED_SMALL_UPLOAD',len(payload),'bytes');break
 except Exception as e:
  if attempt==2:raise SystemExit('Small upload failed: '+type(e).__name__)
  time.sleep(2)
