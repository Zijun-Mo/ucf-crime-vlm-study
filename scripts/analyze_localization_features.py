import csv,json,numpy as np,collections
from scipy.stats import spearmanr
rows=list(csv.DictReader(open('results/diagnostics_localization/per_video_localization_delta.csv')));meta={r['name']:r for r in json.load(open('results/finetuned_test/manifest.json'))['videos']}
for m in ['qwen','openjev']:
 rr=[r for r in rows if r['model']==m]
 for r in rr:
  for k in ['frames','positive_frames','positive_fraction','zero_shot_auroc','mil_auroc','difference']:r[k]=float(r[k])
  x=meta[r['name']];r['duration_s']=r['frames']/x['fps'];r['positive_s']=r['positive_frames']/x['fps'];r['intervals_n']=sum(1 for a,b in x['intervals'] if a>=1 and b>=a)
 print('\n',m)
 for name,fn in [('down',lambda r:r['difference']<0),('up',lambda r:r['difference']>0),('large_drop_10pp',lambda r:r['difference']<=-.1)]:
  v=[r for r in rr if fn(r)];print(name,'n',len(v),'median_fraction',np.median([r['positive_fraction'] for r in v]),'median_video_sec',np.median([r['duration_s'] for r in v]),'median_positive_sec',np.median([r['positive_s'] for r in v]),'median_baseline',np.median([r['zero_shot_auroc'] for r in v]))
 for k in ['positive_fraction','duration_s','positive_s','zero_shot_auroc','intervals_n']:
  print('spearman',k,*(round(float(x),4) for x in spearmanr([r[k] for r in rr],[r['difference'] for r in rr])))
 cats=collections.defaultdict(list)
 for r in rr:cats[r['category']].append(r)
 for k,v in sorted(cats.items(), key=lambda x:-len(x[1])):
  print(k,'n',len(v),'down',sum(r['difference']<0 for r in v),'mean_pp',round(100*np.mean([r['difference'] for r in v]),2))
 print('worst')
 for r in sorted(rr,key=lambda r:r['difference'])[:5]: print(r['name'],round(100*r['zero_shot_auroc'],1),round(100*r['mil_auroc'],1),round(100*r['positive_fraction'],1),round(r['duration_s'],1),round(r['positive_s'],1))
