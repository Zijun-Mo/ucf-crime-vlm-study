"""Fresh Colab SDK assignments; independent bounded reads, no model calls."""
import argparse,base64,json,time
from urllib.parse import quote
import requests
from colab_cli.common import state
p=argparse.ArgumentParser();p.add_argument('endpoint');a=p.parse_args();report={'observed_at':time.time()}
try:
 assignment=next((x for x in state.client.list_assignments() if x.endpoint==a.endpoint),None)
 report['assignment_present']=assignment is not None
 if assignment:
  proxy=assignment.runtime_proxy_info;params={'authuser':'0','colab-runtime-proxy-token':proxy.token}
  def read(path):
   try:
    r=requests.get(proxy.url.rstrip('/')+'/api/contents/'+quote(path.lstrip('/'),safe='/'),params={**params,'content':'1'},timeout=7)
    if r.status_code!=200:return None,r.status_code
    d=r.json();value=d['content'];value=base64.b64decode(value).decode() if d.get('format')=='base64' else value
    return value,200
   except requests.RequestException as e:return None,type(e).__name__
  text,code=read('/content/ucf_localization_train/status.json');report['progress_http_status']=code
  if text:
   d=json.loads(text);keys=['status','stage','steps','epoch_cursor','video_cursor','teacher_videos','baseline','validation','best_epoch','best_localization','best_frame_auroc','bad_epochs','selection_status','last_epoch_active_hours','reason','stop_reason']
   report['progress']={k:d[k] for k in keys if k in d};report['fresh_progress']=True
  text,code=read('/content/ucf_background_handle.json')
  if text:
   handle=json.loads(text);text,code=read('/proc/'+str(handle['pid'])+'/stat')
   if code==404:report['worker_alive']=False
   elif text:
    f=text.split(') ',1)[1].split();report['worker_alive']=f[0]!='Z' and f[19]==handle['start_ticks']
except Exception as e:report['observation_error']=type(e).__name__
print(json.dumps(report,ensure_ascii=False))
