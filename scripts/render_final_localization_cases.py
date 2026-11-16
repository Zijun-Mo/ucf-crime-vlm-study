"""Render deterministic epoch5 case curves using existing predictions only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.study_evaluation import frame_scores
parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=[5,9],default=5);args=parser.parse_args()
experiment='localization_v4_plateau_test' if args.epoch==9 else 'localization_v4_epoch5_test'
policy=json.loads((ROOT/f'research/{experiment}/policy.json').read_text())
records={r['name']:r for r in policy['records']}
dest=ROOT/'results'/('qwen_localization_v4_plateau_test' if args.epoch==9 else 'qwen_localization_v4_epoch5_test')
cases=json.loads((dest/'case_selection.json').read_text())['cases']
arms={'Zero-shot':ROOT/'results/controlled/qwen_zs_p1_4s/predictions',
      'Old MIL':ROOT/'results/controlled/qwen_formal_mil_p1_4s/predictions',
      'Soft E2':ROOT/'results/qwen_localization_v4_epoch2_test/raw/predictions',
      'Soft E5':ROOT/'results/qwen_localization_v4_epoch5_test/raw/predictions'}
if args.epoch==9:arms['Soft E9']=dest/'raw/predictions'
for normalized in [False,True]:
    fig,axes=plt.subplots(3,1,figsize=(11,8.5),layout='constrained')
    for ax,role,title in zip(axes,['worst','median','best'],['Largest drop','Median change','Largest gain']):
        case=cases[role];row=records[case['name']];t=np.arange(row['n_frames'])/row['fps']
        for label,folder in arms.items():
            p=json.loads((folder/(row['name']+'.json')).read_text());assert p['source_sha256']==row['sha256']
            z=frame_scores(row,p['start_s'],p['raw_logits'])
            shown=(z-z.min())/max(float(np.ptp(z)),1e-12) if normalized else z
            ax.plot(t,shown,label=label,linewidth=1.1,alpha=.9)
        for a,b in row['intervals']:
            if a>=1 and b>=a:ax.axvspan((a-1)/row['fps'],b/row['fps'],color='#d64d55',alpha=.13)
        ax.set_title(f'{title}: {row["name"]}; epoch{args.epoch} within-video AUROC delta {100*case["within_delta"]:+.2f} pp',fontsize=10)
        ax.set_xlabel('Time (s)');ax.set_ylabel('Display-normalized score' if normalized else 'Raw frame logit');ax.grid(alpha=.15)
    axes[0].legend(ncol=len(arms),fontsize=9)
    fig.suptitle('Per-video normalization for display only; metrics use original logits' if normalized else f'Frozen epoch{args.epoch} descriptive cases; red = benchmark anomaly interval')
    stem='case_curves_shape' if normalized else 'case_curves_raw'
    for ext in ['png','svg']:fig.savefig(dest/(stem+'.'+ext),dpi=180)
    plt.close(fig)
print(f'Rendered fixed epoch{args.epoch} case curves; no new inference')
