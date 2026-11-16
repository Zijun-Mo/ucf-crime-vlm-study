"""CPU analysis of completed, frozen predictions; never fit or select a model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.learning import atomic_json
from openjev_dss.study_evaluation import frame_labels,frame_scores


def decompose(labels,scores,normal):
    y=np.concatenate(labels);z=np.concatenate(scores);all_auc=float(roc_auc_score(y,z))
    positive=z[y==1];normal_negative=np.concatenate([a for a,n in zip(scores,normal) if n])
    abnormal_background=np.concatenate([a[b==0] for a,b,n in zip(scores,labels,normal) if not n])
    same_mass=same_correct=0.;within=[]
    for t,p in zip(labels,scores):
        a=int(t.sum());b=len(t)-a
        if a and b:
            auc=float(roc_auc_score(t,p));same_mass+=a*b;same_correct+=a*b*auc;within.append(auc)
    total_mass=float(len(positive)*(len(y)-len(positive)));weight=same_mass/total_mass
    def pair_auc(negative):return float(roc_auc_score(np.r_[np.ones(len(positive)),np.zeros(len(negative))],np.r_[positive,negative]))
    threshold=float(np.quantile(positive,.2,method='lower'))
    return dict(pooled_auroc=all_auc,pooled_ap=float(average_precision_score(y,z)),
        within_macro_auroc=float(np.mean(within)),same_video_pair_weight=weight,
        same_video_pair_weighted_auroc=same_correct/same_mass,
        cross_video_pair_auroc=(all_auc*total_mass-same_correct)/(total_mass-same_mass),
        anomaly_vs_normal_video_auroc=pair_auc(normal_negative),
        anomaly_vs_abnormal_background_auroc=pair_auc(abnormal_background),
        normal_video_negative_weight=len(normal_negative)/(len(y)-len(positive)),
        score_groups={name:dict(frames=len(a),mean=float(a.mean()),median=float(np.median(a)),q90=float(np.quantile(a,.9)),
            false_positive_rate_at_80pct_anomaly_recall=float(np.mean(a>=threshold)) if name!='true_anomaly' else None,
            fraction_at_fixed_zero_logit=float(np.mean(a>=0)))
            for name,a in [('normal_video',normal_negative),('abnormal_background',abnormal_background),('true_anomaly',positive)]},
        diagnostic_anomaly_recall=float(np.mean(positive>=threshold)),diagnostic_threshold=threshold,threshold_scope='test-only descriptive operating point, never training/deployment calibration',
        positive_frames=len(positive),negative_frames=len(y)-len(positive))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--include-teacher',action='store_true');parser.add_argument('--include-soft',action='store_true');parser.add_argument('--include-final',action='store_true');parser.add_argument('--include-plateau',action='store_true');args=parser.parse_args()
    policy=json.loads((ROOT/'research/storyboard_context_formal/policy.json').read_text());records=policy['records']
    arms={'zero_shot':(ROOT/'results/controlled/qwen_zs_p1_4s/predictions','raw_logits'),
          'old_mil':(ROOT/'results/controlled/qwen_formal_mil_p1_4s/predictions','raw_logits'),
          'storyboard':(ROOT/'results/storyboard_context_formal_v1/raw/predictions','storyboard')}
    if not arms['storyboard'][0].exists():
        options=list((ROOT/'results/storyboard_context_formal_v1').rglob('predictions'))
        assert len(options)==1;arms['storyboard']=(options[0],'storyboard')
    if args.include_teacher:arms['teacher_constraint']=(ROOT/'results/qwen_localization_v2_epoch1_test/raw/predictions','raw_logits')
    if args.include_soft:arms['minmax_soft']=(ROOT/'results/qwen_localization_v4_epoch2_test/raw/predictions','raw_logits')
    if args.include_final:arms['minmax_soft_epoch5']=(ROOT/'results/qwen_localization_v4_epoch5_test/raw/predictions','raw_logits')
    if args.include_plateau:arms['minmax_soft_epoch9']=(ROOT/'results/qwen_localization_v4_plateau_test/raw/predictions','raw_logits')
    labels=[frame_labels(r) for r in records];normal=[r['category']=='Normal' for r in records];curves={};reports={};per_video=[]
    for arm,(folder,key) in arms.items():
        values=[]
        for row in records:
            p=json.loads((folder/(row['name']+'.json')).read_text());assert p['source_sha256']==row['sha256']
            values.append(frame_scores(row,p['start_s'],p[key]))
        curves[arm]=values;reports[arm]=decompose(labels,values,normal)
    baseline=curves['zero_shot']
    for i,row in enumerate(records):
        y=labels[i]
        if not 0<y.sum()<len(y):continue
        base=float(roc_auc_score(y,baseline[i]));fraction=float(y.mean());duration=row['n_frames']/row['fps']
        per_video.append(dict(name=row['name'],category=row['category'],duration_s=duration,anomaly_fraction=fraction,
            fraction_bin='<=10%' if fraction<=.1 else '10-30%' if fraction<=.3 else '>30%',
            duration_bin='<=60s' if duration<=60 else '60-180s' if duration<=180 else '>180s',
            baseline_within_auroc=base,within_deltas={arm:float(roc_auc_score(y,curves[arm][i])-base) for arm in arms if arm!='zero_shot'}))
    groups={}
    bootstrap=np.random.default_rng(42)
    for field in ['category','fraction_bin','duration_bin']:
        groups[field]={}
        for value in sorted({r[field] for r in per_video}):
            selected=[r for r in per_video if r[field]==value]
            groups[field][value]=dict(videos=len(selected),mean_within_deltas={arm:float(np.mean([r['within_deltas'][arm] for r in selected])) for arm in arms if arm!='zero_shot'},
                exploratory_ci95={arm:(None if len(selected)<2 else dict(zip(['low','high'],np.quantile(bootstrap.choice([r['within_deltas'][arm] for r in selected],size=(1000,len(selected)),replace=True).mean(axis=1),[.025,.975]).tolist()))) for arm in arms if arm!='zero_shot'})
    folder=ROOT/'results/future_work_v1';folder.mkdir(exist_ok=True)
    output=dict(scope='all completed frozen full-test predictions; descriptive analysis, no new inference or selection',
        frames=1111808,videos=290,results=reports,stratified_within_differences=groups,
        decomposition='Pooled AUROC = same-pair weight * pair-weighted within-video AUROC + remaining weight * cross-video pair AUROC; macro within-video AUROC has different weights',
        uncertainty='Paired intervals for primary metrics remain in each formal experiment; decomposition is descriptive; subgroup intervals are exploratory video bootstrap without multiplicity correction, not causal evidence')
    atomic_json(folder/'frame_transfer_analysis.json',output);atomic_json(folder/'per_video_transfer.json',per_video)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16.5,4.2) if args.include_plateau else (15,4.1) if args.include_final else (13,3.9),layout='constrained');names=list(arms);x=np.arange(len(names))
    short=dict(zero_shot='Zero-shot',old_mil='Old MIL',storyboard='Montage',teacher_constraint='Teacher V2',minmax_soft='Soft E2',minmax_soft_epoch5='Soft E5',minmax_soft_epoch9='Soft E9')
    for ax,key,title in zip(axes,['pooled_auroc','pooled_ap','within_macro_auroc'],['Pooled frame AUROC','Pooled frame AP','Within-video macro AUROC']):
        values=[reports[a][key]*100 for a in names];ax.bar(x,values,color=['#4361a5','#db7c26','#5b9a79','#a267ac','#bd4772','#248d97','#4c6841'][:len(names)])
        ax.set_xticks(x,[short[n] for n in names],rotation=20,ha='right');ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(values):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
    fig.suptitle('Frozen Qwen experiments: detection and localization (point estimates)')
    fig.savefig(folder/'frame_transfer.png',dpi=180);fig.savefig(folder/'frame_transfer.svg');plt.close(fig)
    reference=reports['zero_shot'];variants=[a for a in arms if a!='zero_shot'];w=reference['normal_video_negative_weight']
    contribution={a:dict(normal_video_comparison_pp=100*w*(reports[a]['anomaly_vs_normal_video_auroc']-reference['anomaly_vs_normal_video_auroc']),
        abnormal_background_comparison_pp=100*(1-w)*(reports[a]['anomaly_vs_abnormal_background_auroc']-reference['anomaly_vs_abnormal_background_auroc']),
        total_pooled_auroc_change_pp=100*(reports[a]['pooled_auroc']-reference['pooled_auroc'])) for a in variants}
    atomic_json(folder/'auc_contributions.json',dict(negative_frame_weights=dict(normal_videos=w,abnormal_video_background=1-w),
        contributions=contribution,scope='exact descriptive decomposition, not causal attribution'))
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained');xx=np.arange(len(variants));width=.32
    ax.bar(xx-width/2,[contribution[a]['normal_video_comparison_pp'] for a in variants],width,label='Anomaly vs normal-video frames',color='#438b6b')
    ax.bar(xx+width/2,[contribution[a]['abnormal_background_comparison_pp'] for a in variants],width,label='Anomaly vs abnormal-video background',color='#cd753b')
    ax.scatter(xx,[contribution[a]['total_pooled_auroc_change_pp'] for a in variants],color='black',label='Total pooled AUROC change',zorder=5)
    ax.axhline(0,color='grey',linewidth=.8);ax.set_xticks(xx,variants);ax.set_ylabel('Contribution to AUROC change (percentage points)');ax.set_title('Which frame comparisons produce the gain?');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'frame_auc_decomposition.png',dpi=180);fig.savefig(folder/'frame_auc_decomposition.svg');plt.close(fig)
    # Hold the same descriptive cases across follow-up methods. Adding a method
    # must not silently change the visible examples or their image evidence.
    selection_arm='old_mil';ordered=sorted(per_video,key=lambda r:(r['within_deltas'][selection_arm],r['name']))
    selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    atomic_json(folder/'descriptive_case_selection.json',dict(comparison=selection_arm,rule='min/median/max within-video AUROC change; descriptive only',cases=selected))
    fig,axes=plt.subplots(3,1,figsize=(11,8.2),layout='constrained')
    for ax,case,label in zip(axes,selected,['Largest drop','Median change','Largest gain']):
        i=next(j for j,r in enumerate(records) if r['name']==case['name']);row=records[i];t=np.arange(row['n_frames'])/row['fps']
        for arm in arms:ax.plot(t,curves[arm][i],label=arm,linewidth=1.1,alpha=.85)
        for a,b in row['intervals']:
            if a>=1 and b>=a:ax.axvspan((a-1)/row['fps'],b/row['fps'],color='#d64d55',alpha=.12)
        ax.set_title(f'{label}: {row["name"]}, within delta {100*case["within_deltas"][selection_arm]:+.2f} pp',fontsize=10)
        ax.set_xlabel('Time (s)');ax.set_ylabel('Raw frame logit');ax.grid(alpha=.15)
    axes[0].legend(ncol=len(arms),fontsize=8);fig.suptitle('Fixed descriptive cases; red bands show benchmark anomaly intervals')
    fig.savefig(folder/'descriptive_cases.png',dpi=170);fig.savefig(folder/'descriptive_cases.svg');plt.close(fig)
    if args.include_soft:
        own=sorted(per_video,key=lambda r:(r['within_deltas']['minmax_soft'],r['name']))
        chosen=[own[0],own[len(own)//2],own[-1]]
        atomic_json(folder/'soft_case_selection.json',dict(rule='min/median/max within-video AUROC change vs zero shot, deterministic filename ties',cases=chosen))
        fig,axes=plt.subplots(3,1,figsize=(11,8.2),layout='constrained')
        for ax,case,label in zip(axes,chosen,['Largest drop','Median change','Largest gain']):
            i=next(j for j,r in enumerate(records) if r['name']==case['name']);row=records[i];t=np.arange(row['n_frames'])/row['fps']
            for arm in ['zero_shot','old_mil','teacher_constraint','minmax_soft']:
                if arm in arms:ax.plot(t,curves[arm][i],label=arm,linewidth=1.1,alpha=.85)
            for a,b in row['intervals']:
                if a>=1 and b>=a:ax.axvspan((a-1)/row['fps'],b/row['fps'],color='#d64d55',alpha=.12)
            ax.set_title(f'{label}: {row["name"]}, within delta {100*case["within_deltas"]["minmax_soft"]:+.2f} pp',fontsize=10)
            ax.set_xlabel('Time (s)');ax.set_ylabel('Raw frame logit');ax.grid(alpha=.15)
        axes[0].legend(ncol=4,fontsize=8);fig.suptitle('Frozen min-max soft-label cases; red = benchmark anomaly intervals')
        fig.savefig(folder/'minmax_soft_cases.png',dpi=180);fig.savefig(folder/'minmax_soft_cases.svg');plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(11,8.2),layout='constrained')
        for ax,case,label in zip(axes,chosen,['Largest drop','Median change','Largest gain']):
            i=next(j for j,r in enumerate(records) if r['name']==case['name']);row=records[i];t=np.arange(row['n_frames'])/row['fps']
            for arm in ['zero_shot','old_mil','teacher_constraint','minmax_soft']:
                if arm not in arms:continue
                z=curves[arm][i];shown=(z-z.min())/max(float(np.ptp(z)),1e-12)
                ax.plot(t,shown,label=arm,linewidth=1.1,alpha=.85)
            for a,b in row['intervals']:
                if a>=1 and b>=a:ax.axvspan((a-1)/row['fps'],b/row['fps'],color='#d64d55',alpha=.12)
            ax.set_title(f'{label}: {row["name"]}, within delta {100*case["within_deltas"]["minmax_soft"]:+.2f} pp',fontsize=10)
            ax.set_xlabel('Time (s)');ax.set_ylabel('Display-normalized score');ax.grid(alpha=.15)
        axes[0].legend(ncol=4,fontsize=8);fig.suptitle('Per-video normalization for display only; all metrics use original logits')
        fig.savefig(folder/'minmax_soft_cases_shape.png',dpi=180);fig.savefig(folder/'minmax_soft_cases_shape.svg');plt.close(fig)
    print(json.dumps({a:{k:r[k] for k in ['pooled_auroc','pooled_ap','within_macro_auroc','same_video_pair_weight','cross_video_pair_auroc']} for a,r in reports.items()},indent=2))
if __name__=='__main__':main()
