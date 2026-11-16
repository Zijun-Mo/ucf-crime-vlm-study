"""Resumable CPU preparation with bounded concurrent download/decode."""
import json,hashlib,math,shutil,time,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

def prepare(root,full=False,budget_seconds=28800,workers=4):
 import cv2,requests
 from requests.adapters import HTTPAdapter
 from urllib3.util.retry import Retry
 from .audit import canonical_phashes
 cv2.setNumThreads(1);root=Path(root);m=json.loads((root/'manifest.json').read_text())
 for d in ['sources','canonical','metadata']:(root/d).mkdir(exist_ok=True)
 work=Path('/content/dss_cpu');work.mkdir(exist_ok=True)
 lock=threading.Lock();stop=threading.Event();state={'status':'running','started':time.time(),'completed':[],'inflight':[],'workers':workers}
 last_save=-float('inf')
 def persist(force=False):
  nonlocal last_save
  if not force and time.monotonic()-last_save<5:return
  last_save=time.monotonic()
  tmp=root/'cpu_status.tmp';tmp.write_text(json.dumps(state));tmp.replace(root/'cpu_status.json')
 def guard():
  if stop.is_set() or time.time()-state['started']>budget_seconds:raise TimeoutError('CPU preparation allocation limit or task stopped')
 def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  return h.hexdigest()
 def process(rec):
  guard();name=rec['name'];meta=root/'metadata'/(name+'.json');dst=root/'canonical'/name
  with lock:state['inflight'].append(name);persist()
  try:
   if meta.exists() and dst.exists():
    info=json.loads(meta.read_text());assert info['sha256']==rec['sha256']
    if info.get('fingerprint_version') not in ['canonical-uniform8-v2','canonical-uniform-plus30s-v3']:
     info['phashes']=canonical_phashes(dst,info['sampled_frames']);info['fingerprint_version']='canonical-uniform8-v2';meta.write_text(json.dumps(info))
   else:
    src=root/'sources'/name;local=work/name
    if src.exists():shutil.copyfile(src,local)
    else:
     session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
     url=f"https://huggingface.co/datasets/{m['dataset']}/resolve/{m['revision']}/{rec['repo_path']}"
     with session.get(url,stream=True,timeout=(30,120)) as r:
      r.raise_for_status()
      with local.open('wb') as f:
       for b in r.iter_content(4*1024*1024):guard();f.write(b)
     session.close()
    assert sha(local)==rec['sha256'],name
    if not src.exists():
     temp=src.with_suffix('.partial');shutil.copyfile(local,temp);temp.replace(src)
    cap=cv2.VideoCapture(str(local));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>=2 and n>0
    wanted={math.floor(j*fps/2) for j in range(math.ceil(2*n/fps))}
    target=work/('canonical_'+name);enc=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),2,(224,224));assert enc.isOpened();sampled=0
    try:
     for i in range(n):
      if i%300==0:guard()
      assert cap.grab(),(name,i)
      if i in wanted or i==n-1:
       ok,f=cap.retrieve();assert ok;small=cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA)
       if i in wanted:enc.write(small);sampled+=1
       if i==n-1:pad=small
     assert not cap.grab();enc.write(pad)
    finally:enc.release();cap.release()
    hashes=canonical_phashes(target,sampled)
    digest=sha(target);temp=dst.with_suffix('.partial');shutil.copyfile(target,temp);temp.replace(dst)
    target.unlink();local.unlink()
    info={**rec,'n_frames':n,'fps':fps,'window_count':math.ceil(n/fps),'sampled_frames':sampled,'canonical_sha256':digest,'phashes':hashes,'fingerprint_version':'canonical-uniform8-v2'}
    temp=meta.with_suffix('.tmp');temp.write_text(json.dumps(info));temp.replace(meta)
   with lock:state['completed'].append(name)
  finally:
   with lock:state['inflight'].remove(name);persist()
 try:
  with ThreadPoolExecutor(max_workers=workers) as executor:
   futures=[executor.submit(process,r) for r in (m['records'] if full else m['smoke'])]
   for f in as_completed(futures):
    try:f.result()
    except Exception:
     stop.set()
     for pending in futures:pending.cancel()
     raise
  state.update(status='complete',elapsed_s=time.time()-state['started'],source_bytes=sum(p.stat().st_size for p in (root/'sources').glob('*.mp4')),canonical_bytes=sum(p.stat().st_size for p in (root/'canonical').glob('*.mp4')))
 except Exception:state.update(status='error',error=traceback.format_exc())
 persist(force=True);print(json.dumps({k:v for k,v in state.items() if k!='completed'}),flush=True)
