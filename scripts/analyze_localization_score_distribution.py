import json,numpy as np
from pathlib import Path
from openjev_dss.study_evaluation import frame_labels,frame_scores
from collections import defaultdict
m=json.load(open('results/finetuned_test/manifest.json'));d={r['name']:r for r in m['videos']};out={}
for model in ['qwen','openjev']:
 reports=[]
 for variant in ['zs','formal_mil']:
  p=f'results/controlled/{model}_{variant}_p1_4s/predictions';items=[]
  for n,r in d.items():
   y=frame_labels(r);yn=int(y.sum());
   if yn<=0 or yn==len(y):continue
   j=json.load(open(Path(p)/(n+'.json')));z=frame_scores(r,j['start_s'],np.asarray(j['raw_logits']),4,1);k=max(1,int(np.ceil(len(y)*.1)));threshold=np.partition(z,-k)[-k];above=z>threshold;tied=z==threshold;hit=(y[above].sum()+((k-above.sum())/tied.sum())*y[tied].sum())/k
   sd=z.std();margin=(z[y==1].mean()-z[y==0].mean())/sd if sd else 0
   items.append((n,float(hit),float(margin)))
  reports.append({n:(p,m) for n,p,m in items})
 delta=[reports[1][n][0]-v[0] for n,v in reports[0].items()];md=[reports[1][n][1]-v[1] for n,v in reports[0].items()];print(model,'n',len(delta),'top10 anomaly frac zs/mil',np.mean([x[0] for x in reports[0].values()]),np.mean([x[0] for x in reports[1].values()]),'paired diff',np.mean(delta),'margin zs/mil',np.mean([x[1] for x in reports[0].values()]),np.mean([x[1] for x in reports[1].values()]),'margin delta',np.mean(md),'d count',sum(x<0 for x in md))
