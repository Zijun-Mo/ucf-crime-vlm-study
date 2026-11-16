"""Analyze selected descriptive cases using fixed P1/4s predictions and frame labels."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from openjev_dss.study_evaluation import frame_labels,frame_scores
cases=[('qwen','Stealing019_x264.mp4'),('qwen','RoadAccidents009_x264.mp4'),('openjev','Shooting013_x264.mp4'),('openjev','Explosion004_x264.mp4')]
records={r['name']:r for r in json.loads(Path('results/finetuned_test/manifest.json').read_text())['videos']};out=Path('results/diagnostics_localization/cases');out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(4,1,figsize=(13,11),constrained_layout=True);results={}
for ax,(model,n) in zip(axes,cases):
 r=records[n];y=frame_labels(r);fps=r['fps'];entry={'model':model,'annotation_seconds':[[round((a-1)/fps,3),round(b/fps,3)] for a,b in r['intervals'] if a>=1 and b>=a],'n_frames':len(y),'fps':fps,'variants':{}}
 for tag,label,color in [('zs','zero-shot','#3772a8'),('formal_mil','formal MIL','#d24a33')]:
  p=json.loads(Path(f'results/controlled/{model}_{tag}_p1_4s/predictions/{n}.json').read_text());v=np.asarray(p['raw_logits']);z=frame_scores(r,p['start_s'],v,4,1);steps=np.unique(np.r_[np.arange(0,len(y),max(1,round(fps))),len(y)-1]);norm=(z-z.min())/(z.max()-z.min()) if z.max()>z.min() else np.zeros_like(z);ax.plot(steps/fps,norm[steps],label=label,color=color,lw=1.5)
  k=max(1,math.ceil(len(y)*.1));threshold=np.partition(z,-k)[-k];hi=z>threshold;ties=z==threshold;tp=(y[hi].sum()+(k-hi.sum())/ties.sum()*y[ties].sum())/k;i=int(np.argmax(z));entry['variants'][tag]={'within_auroc':float(roc_auc_score(y,z)),'mean_positive_raw_logit':float(z[y==1].mean()),'mean_negative_raw_logit':float(z[y==0].mean()),'max_frame_score_time_s':i/fps,'max_frame_score_is_annotated':bool(y[i]),'highest_window_start_s':p['start_s'][int(np.argmax(v))],'top10_frame_anomaly_fraction_tie_adjusted':float(tp)}
 for a,b in entry['annotation_seconds']:ax.axvspan(a,b,color='gray',alpha=.25)
 ax.set_title(f'{n} — {model}');ax.set_ylabel('Per-curve min–max\n(normalized)');ax.set_xlabel('time (seconds); gray = annotated anomaly');ax.set_ylim(-.04,1.04);ax.grid(alpha=.15);ax.legend(loc='upper right');results[n]=entry
fig.savefig(out/'case_score_timelines.png',dpi=150);fig.savefig(out/'case_score_timelines.svg');(out/'case_score_statistics.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
