"""Post-selection, paired-video contrasts between existing complete predictions."""
import argparse,json,sys,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.study_evaluation import evaluate,interval,resampling_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=[2,5,9],default=2);args=parser.parse_args()
    experiment={2:'localization_v4_test',5:'localization_v4_epoch5_test',9:'localization_v4_plateau_test'}[args.epoch]
    dest=ROOT/'results'/('qwen_localization_v4_plateau_test' if args.epoch==9 else f'qwen_localization_v4_epoch{args.epoch}_test')
    policy=json.loads((ROOT/f'research/{experiment}/policy.json').read_text());records=policy['records']
    folders=dict(old_mil=ROOT/'results/controlled/qwen_formal_mil_p1_4s/predictions',
        teacher_constraint=ROOT/'results/qwen_localization_v2_epoch1_test/raw/predictions',
        minmax_soft=dest/'raw/predictions')
    if args.epoch in [5,9]:folders['minmax_soft_epoch2']=ROOT/'results/qwen_localization_v4_epoch2_test/raw/predictions'
    if args.epoch==9:folders['minmax_soft_epoch5']=ROOT/'results/qwen_localization_v4_epoch5_test/raw/predictions'
    weights=resampling_weights(290,1000,42);reports={};samples={}
    for arm,folder in folders.items():
        predictions={}
        for row in records:
            p=json.loads((folder/(row['name']+'.json')).read_text());assert p['source_sha256']==row['sha256']
            predictions[row['name']]={**p,'score':p['raw_logits']}
        reports[arm],samples[arm]=evaluate(records,predictions,'score',weights)
    official=dict(old_mil=json.loads((ROOT/'results/study_formal_v2/qwen_formal_mil_p1_4s.json').read_text())['metrics'],
        teacher_constraint=json.loads((ROOT/'results/qwen_localization_v2_epoch1_test/summary.json').read_text())['reports']['candidate']['metrics'],
        minmax_soft=json.loads((dest/'summary.json').read_text())['reports']['candidate']['metrics'])
    if args.epoch in [5,9]:official['minmax_soft_epoch2']=json.loads((ROOT/'results/qwen_localization_v4_epoch2_test/summary.json').read_text())['reports']['candidate']['metrics']
    if args.epoch==9:official['minmax_soft_epoch5']=json.loads((ROOT/'results/qwen_localization_v4_epoch5_test/summary.json').read_text())['reports']['candidate']['metrics']
    for arm in reports:
        for key,value in reports[arm]['metrics'].items():assert abs(value['value']-official[arm][key]['value'])<1e-12
    contrasts={}
    for other in [a for a in folders if a!='minmax_soft']:
        contrasts['minmax_soft_minus_'+other]={k:dict(difference=reports['minmax_soft']['metrics'][k]['value']-reports[other]['metrics'][k]['value'],
            ci95=interval(samples['minmax_soft'][k]-samples[other][k])) for k in samples['minmax_soft']}
    result=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),videos=290,bootstrap=dict(repeats=1000,seed=42,unit='paired video'),
        metrics={a:{k:v['value'] for k,v in r['metrics'].items()} for a,r in reports.items()},
        contrasts=contrasts,post_selection_descriptive_comparisons=True,multiplicity_adjusted=False,new_inference=False,
        equivalence_test=False,selected_epoch=args.epoch,limitation='Close point estimates do not establish equivalence; no contribution of an individual loss term can be isolated without ablation')
    (dest/'method_comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=json.loads((dest/'summary.json').read_text());keys=['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap']
    labels=['Pooled frame AUROC','Pooled frame AP','Within-video macro AUROC','Video max AP']
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    for i,k in enumerate(keys):
        d=main['paired_contrasts'][k];point=100*d['difference'];lo=100*d['ci95']['low'];hi=100*d['ci95']['high']
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4,color='#286b64' if lo>0 else '#7e647f')
    ax.axvline(0,color='grey',linewidth=.9);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('Difference vs zero-shot baseline (percentage points)')
    ax.set_title(f'Frozen min-max soft-label model, epoch {args.epoch}: paired 95% intervals');ax.grid(axis='x',alpha=.2)
    fig.savefig(dest/'paired_differences.png',dpi=200);fig.savefig(dest/'paired_differences.svg');plt.close(fig)
    print(json.dumps({a:{k:v for k,v in d.items() if k in keys} for a,d in contrasts.items()},indent=2))


if __name__=='__main__':main()
