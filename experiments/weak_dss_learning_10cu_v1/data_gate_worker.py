import os,sys,time,json,traceback
from pathlib import Path
ROOT=Path('/content/drive/MyDrive/OpenJev_UCFCrime');OUT=ROOT/'weak_dss_learning_10cu_v1';DATA=ROOT/'weak_dss_smoke_v1'
state={'status':'waiting_for_preparation','started':time.time(),'test_temporal_labels_used':False}
def persist():(OUT/'data_gate_status.json').write_text(json.dumps(state))
try:
 persist()
 while True:
  prep=json.loads((DATA/'cpu_status.json').read_text())
  if prep['status']=='error':raise RuntimeError(prep.get('error'))
  if prep['status']=='complete' and len(prep['completed'])==1602:break
  if time.time()-state['started']>28800:raise TimeoutError('Data preparation time limit')
  time.sleep(15)
 from openjev_dss.audit import make_test_fingerprints,refresh_dense_fingerprints
 from openjev_dss.learning_audit import isolate
 state['status']='dense_image_fingerprints';persist();refresh_dense_fingerprints(DATA)
 state['status']='image_audit';persist()
 m=json.loads((DATA/'manifest.json').read_text());fp=OUT/'test_image_fingerprints.json'
 if not fp.exists():make_test_fingerprints(ROOT/'formal_stride1',m['test_identity'],fp)
 counts,report=isolate(DATA,fp,OUT)
 m=json.loads((OUT/'manifest.json').read_text());duration={}
 for sp in ['train','val','cal']:
  rs=[r for r in m['records'] if r['split']==sp]
  metas=[json.loads((DATA/'metadata'/(r['name']+'.json')).read_text()) for r in rs]
  duration[sp]={'videos':len(rs),'full_windows':sum(r['window_count'] for r in metas),'epoch_sample_windows':sum(min(16,r['window_count']) for r in metas),'normal':sum(not r['label'] for r in rs),'abnormal':sum(r['label'] for r in rs)}
 state.update(status='complete',counts=counts,durations=duration,new_quarantined=len(report['new_quarantine']),candidate_pairs=len(report['candidates']))
except Exception:state.update(status='error',error=traceback.format_exc())
finally:
 state['elapsed_s']=time.time()-state['started'];persist()
 print(json.dumps(state),flush=True)
