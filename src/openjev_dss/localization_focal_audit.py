"""CPU-only qualification of frozen instance supervision and frame validation."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from .study_evaluation import frame_labels,frame_scores
from .localization_focal import select
from .localization_pointwise import coefficients


def development_metrics(records,scores,annotations,metadata):
    truth=[];pred=[];names=[]
    for row in records:
        name=row['name']
        if row['label'] and name not in annotations:continue
        meta=annotations[name] if row['label'] else {**metadata[name],'intervals':[]}
        z=scores[name];truth.append(frame_labels(meta));pred.append(frame_scores(meta,list(range(len(z))),z));names.append(name)
    y=np.concatenate(truth);z=np.concatenate(pred)
    return dict(dev_pooled_auroc=float(roc_auc_score(y,z)),dev_pooled_ap=float(average_precision_score(y,z)),
        dev_frame_videos=len(names),dev_frame_count=len(y),dev_frame_names=names,
        dev_scope='reviewed validation anomalies plus all normal validation videos; unannotated anomalies excluded')


def pseudo_label_audit(annotations,scores,reference,config):
    totals=dict(positive_mass=0.,positive_correct_mass=0.,background_mass=0.,background_correct_mass=0.,
        positive_any_overlap_mass=0.,background_clean_window_mass=0.,positive_oracle_purity_mass=0.)
    rows=[]
    for name,meta in annotations.items():
        y=frame_labels(meta);selected=select(scores[name],reference,config)
        fractions=[]
        for i in range(len(scores[name])):
            start=min(len(y)-1,int(round(i*meta['fps'])))
            end=min(len(y),int(round((i+4)*meta['fps'])))
            fractions.append(float(y[start:max(start+1,end)].mean()))
        f=np.asarray(fractions);a=coefficients(selected);b=np.asarray(selected['background'])
        row=dict(name=name,positive_mass=float(a.sum()),positive_correct_mass=float(a@f),
            background_mass=float(b.sum()),background_correct_mass=float(b@(1-f)),
            positive_any_overlap_mass=float(a@(f>0)),background_clean_window_mass=float(b@(f==0)),
            positive_oracle_purity_mass=float(a.sum()*f.max()),oracle_max_anomaly_fraction=float(f.max()))
        for key in totals:totals[key]+=row[key]
        rows.append(row)
    positive=totals['positive_correct_mass']/totals['positive_mass'] if totals['positive_mass'] else None
    background=totals['background_correct_mass']/totals['background_mass'] if totals['background_mass'] else None
    passed=positive is not None and background is not None and positive>=.8 and background>=.9
    return dict(passed=passed,positive_anomaly_overlap=positive,background_normal_overlap=background,
        positive_any_overlap_rate=totals['positive_any_overlap_mass']/totals['positive_mass'] if totals['positive_mass'] else None,
        background_clean_window_rate=totals['background_clean_window_mass']/totals['background_mass'] if totals['background_mass'] else None,
        positive_oracle_purity_upper=totals['positive_oracle_purity_mass']/totals['positive_mass'] if totals['positive_mass'] else None,
        positive_threshold=.8,background_threshold=.9,videos=len(rows),totals=totals,per_video=rows,
        supervision='validation annotation audit only; no temporal labels enter training',
        interpretation='Overlap thresholds measure frame purity inside4s windows, not window-label accuracy; any-overlap and clean-window rates are separate descriptive diagnostics and cannot override the frozen gate',test_used=False)
