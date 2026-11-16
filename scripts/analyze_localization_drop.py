"""Descriptive paired video-level localization change; no model fitting."""
import json,csv
from pathlib import Path
import numpy as np
root=Path('results/study_formal_v2'); out=Path('results/diagnostics_localization');out.mkdir(exist_ok=True,parents=True)
rows=[];stats={}
for model in ('qwen','openjev'):
 a={r['name']:r for r in json.loads((root/f'{model}_zs_p1_4s.json').read_text())['per_video']};b={r['name']:r for r in json.loads((root/f'{model}_formal_mil_p1_4s.json').read_text())['per_video']};assert a.keys()==b.keys()
 data=[]
 for name,x in a.items():
  y=b[name];assert (x['category'],x['frames'],x['positive_frames'])==(y['category'],y['frames'],y['positive_frames'])
  if x['within_auroc'] is None:assert y['within_auroc'] is None;continue
  row={'model':model,'name':name,'category':x['category'],'frames':x['frames'],'positive_frames':x['positive_frames'],'positive_fraction':x['positive_frames']/x['frames'],'zero_shot_auroc':x['within_auroc'],'mil_auroc':y['within_auroc'],'difference':y['within_auroc']-x['within_auroc']};data.append(row);rows.append(row)
 d=np.array([r['difference'] for r in data]);groups={}
 for category in sorted({r['category'] for r in data}):
  v=[r['difference'] for r in data if r['category']==category];groups[category]={'n':len(v),'mean_difference':float(np.mean(v))}
 bins=[]
 for lo,hi in [(0,.1),(.1,.3),(.3,.6),(.6,1.01)]:
  v=[r['difference'] for r in data if lo<=r['positive_fraction']<hi];bins.append({'lo':lo,'hi':hi,'n':len(v),'mean_difference':float(np.mean(v)) if v else None})
 stats[model]={'eligible':len(d),'declined':int((d<0).sum()),'increased':int((d>0).sum()),'ties':int((d==0).sum()),'mean_difference':float(np.mean(d)),'median_difference':float(np.median(d)),'mean_without_five_largest_drops':float(np.mean(np.sort(d)[5:])),'category':groups,'positive_fraction_bins':bins}
q={r['name']:r['difference'] for r in rows if r['model']=='qwen'};o={r['name']:r['difference'] for r in rows if r['model']=='openjev'};assert q.keys()==o.keys();stats['both_decreased']=sum(q[n]<0 and o[n]<0 for n in q)
with (out/'per_video_localization_delta.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(out/'localization_delta_summary.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
