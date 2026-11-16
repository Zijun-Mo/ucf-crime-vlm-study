"""Export actual multi-epoch training curves and validation-only selection evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.convergence import record_validation

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt',type=Path)
args=parser.parse_args()
if args.receipt:
    receipt_path=args.receipt
else:
    candidates=[p for p in sorted((ROOT/'research/runs').glob('train_formal_*/receipt.json')) if (p.parent/'result.json').exists()]
    if not candidates:raise SystemExit('No actual formal training result exists; do not create synthetic curves')
    receipt_path=candidates[-1]
state=json.loads((receipt_path.parent/'result.json').read_text())
assert state['test_used'] is False
policy=json.loads((ROOT/'research/training_protocol_v2.json').read_text())
out=ROOT/'results/formal_training_v2';out.mkdir(exist_ok=True)
rows=[]
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
summary={}
for name,arm in state['arms'].items():
    replay={}
    for v in arm['validation']:record_validation(replay,v['epoch'],v['video_max_ap'],policy['selection'])
    if arm['validation']:
        assert replay['best_epoch']==arm['best_epoch'] and replay['best_ap']==arm['best_ap']
    epochs=sorted({r['epoch'] for r in arm['rows']})
    losses=[]
    for epoch in epochs:
        steps=[r for r in arm['rows'] if r['epoch']==epoch]
        losses.append(float(np.mean([r['loss'] for r in steps])))
        val=next((v for v in arm['validation'] if v['epoch']==epoch),{})
        rows.append({'branch':name,'epoch':epoch,'steps':len(steps),'mean_loss':losses[-1],
            'mean_grad_norm':float(np.mean([r['grad_norm'] for r in steps])),
            'validation_ap':val.get('video_max_ap'),'validation_auroc':val.get('video_max_auroc'),
            'validation_videos':val.get('videos'),'selected_best':epoch==arm.get('best_epoch')})
    axes[0].plot(epochs,losses,'o-',label=name)
    for ax,key in zip(axes[1:],['video_max_ap','video_max_auroc']):
        axes_x=[v['epoch'] for v in arm['validation']]
        axes_y=[v[key] for v in arm['validation']]
        ax.plot(axes_x,axes_y,'o-',label=name)
        if axes_y:ax.scatter(arm['best_epoch'],next(v[key] for v in arm['validation'] if v['epoch']==arm['best_epoch']),marker='*',s=130)
    summary[name]={'trained_steps':arm['steps'],'fully_validated_epochs':len(arm['validation']),
                  'best_epoch':arm.get('best_epoch'),'best_validation_ap':arm.get('best_ap'),
                  'stop_reason':arm.get('stop_reason'),'frozen_parameters_unchanged':arm.get('frozen_parameters_unchanged',False),
                  'selected_artifact':arm.get('selected_artifact')}
for ax,title in zip(axes,['Training MIL loss','Validation video-max AP','Validation video-max AUROC']):
    ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Formal paired MIL · best checkpoint selected on validation AP · test excluded')
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(out/('learning_curves.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
with (out/'epochs.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['branch','epoch','steps','mean_loss','mean_grad_norm','validation_ap','validation_auroc','validation_videos','selected_best'])
    writer.writeheader();writer.writerows(rows)
evidence={'status':state['status'],'test_used':False,'training_receipt':str(receipt_path.relative_to(ROOT)),
          'training_status_sha256':hashlib.sha256((receipt_path.parent/'result.json').read_bytes()).hexdigest(),
          'policy':policy,'branches':summary,'weights_location':'OpenJev_UCFCrime/remaining_20261108_v1/formal_v2/training/'}
(out/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
lines=['# 正式多轮 MIL 训练','',f"状态：`{state['status']}`。测试集未参与停止或检查点选择。",'',
       '|分支|已训练步数|完整验证轮数|最佳轮次|最佳验证 AP|停止原因|','|---|---:|---:|---:|---:|---|']
for name,s in summary.items():
    ap='待验证' if s['best_validation_ap'] is None else f"{s['best_validation_ap']:.4%}"
    lines.append(f"|{name}|{s['trained_steps']}|{s['fully_validated_epochs']}|{s['best_epoch']}|{ap}|{s['stop_reason']}|")
lines+=['','按预先固定的验证集 AP 规则早停；轮数上限和运行时到期均不代表收敛。',
        '最佳检查点可能来自第 1 轮；后续训练与完整验证的作用是确认早停条件，而非强制选择较晚检查点。',
        '两条分支使用相同数据、优化器设置和停止规则；报告各自实际消耗的轮数。',
        '权重及冻结视觉缓存保留在云盘，位置见 summary.json。','','![训练与验证曲线](learning_curves.png)','']
(out/'README.md').write_text('\n'.join(lines))
print(json.dumps({'status':state['status'],'branches':summary},indent=2))
