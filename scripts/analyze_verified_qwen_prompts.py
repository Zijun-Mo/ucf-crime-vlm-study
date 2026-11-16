"""Analyze the completed Qwen prompt subset without declaring the full study complete."""
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
from openjev_dss.study_evaluation import evaluate,interval,resampling_weights
from openjev_dss.producer_audit import archived_producer_signature

protocol=json.loads((ROOT/'research/protocol.json').read_text())
mp=ROOT/'results/formal_stride1/manifest.json'
assert hashlib.sha256(mp.read_bytes()).hexdigest()==protocol['test_manifest_sha256']
records=json.loads(mp.read_text())['videos']
canonical={r['name']:r for r in json.loads((ROOT/'results/formal_stride1/config.json').read_text())['canonical']}
proof=json.loads((ROOT/'research/recovered_score_integrity.json').read_text())
producer_path=ROOT/proof['controller_receipt'];producer=json.loads(producer_path.read_text())
assert json.loads((producer_path.parent/'environment.json').read_text())['versions']==json.loads((ROOT/'research/gpu_preflight_result.json').read_text())['versions']
weights=resampling_weights(len(records),protocol['evaluation']['bootstrap'],protocol['evaluation']['seed'])
out=ROOT/'results/qwen_prompt_interim';out.mkdir(exist_ok=True)
reports={};samples={};signature=None
for prompt in ['p1','p2','p3']:
    name=f'qwen_zs_{prompt}_4s';folder=ROOT/'results/controlled'/name
    config=json.loads((folder/'config.json').read_text());fp=config.pop('fingerprint')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==fp
    assert proof['conditions'][name]['status']=='passed' and proof['conditions'][name]['fingerprint']==fp
    assert config['protocol_sha256']==hashlib.sha256((ROOT/'research/protocol.json').read_bytes()).hexdigest()
    current=archived_producer_signature(ROOT,config,producer)
    if signature is None:signature=current
    assert signature==current
    assert json.loads((folder/'complete.json').read_text())=={'complete':True,'fingerprint':fp,'videos':290,'windows':37188,'frames':1111808}
    predictions={p.stem:json.loads(p.read_text()) for p in (folder/'predictions').glob('*.json')}
    for r in records:
        p=predictions[r['name']]
        assert p['source_sha256']==r['sha256'] and p['fingerprint']==fp
        assert p['canonical_sha256']==canonical[r['name']]['canonical_sha256']
    report,boot=evaluate(records,predictions,'raw_logits',weights,duration=4)
    assert len(report['eligible_within_videos'])==140 and sum(r['windows'] for r in report['per_video'])==37188
    report.update(condition=name,fingerprint=fp,scope='Completed Qwen prompt subset only; full formal study remains incomplete')
    reports[name]=report;samples[name]=boot
    (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('ANALYZED',name,flush=True)
contrasts={}
for prompt in ['p2','p3']:
    name=f'qwen_zs_{prompt}_4s';base='qwen_zs_p1_4s'
    contrasts[prompt+'_minus_p1']={k:{'difference':reports[name]['metrics'][k]['value']-reports[base]['metrics'][k]['value'],
        'ci95':interval(samples[name][k]-samples[base][k])} for k in samples[name]}
summary={'status':'complete_for_verified_qwen_prompt_subset_only','full_study_complete':False,
    'conditions':list(reports),'bootstrap_replicates':len(weights),'bootstrap_seed':protocol['evaluation']['seed'],
    'paired_contrasts':contrasts,'producer_receipt':proof['controller_receipt'],
    'protocol_sha256':hashlib.sha256((ROOT/'research/protocol.json').read_bytes()).hexdigest(),
    'multiplicity_note':'Pairwise 95% intervals are unadjusted for multiple comparisons; do not infer a broad superiority claim',
    'selection_policy':'Prompts and primary P1 were fixed before these scores; no prompt or training-stopping rule is changed by this analysis',
    'test_history':'Official test set was used in historical exploration; this is not an independent blind test'}
(out/'comparisons.json').write_text(json.dumps(summary,indent=2)+'\n')
metrics=['video_max_auroc','within_macro_auroc','pooled_auroc'];fig,axes=plt.subplots(1,3,figsize=(11,3.8))
for ax,key,title in zip(axes,metrics,['Between-video AUROC','Within-video macro AUROC','Pooled frame AUROC']):
    for i,(name,report) in enumerate(reports.items()):
        row=report['metrics'][key];v=100*row['value'];ci=row['ci95']
        ax.errorbar(i,v,yerr=[[max(0,v-100*ci['low'])],[max(0,100*ci['high']-v)]],fmt='o',capsize=4,color='#336699')
    ax.set_xticks(range(3),['P1','P2','P3']);ax.set_title(title);ax.set_ylabel('AUROC (%)');ax.grid(axis='y',alpha=.2)
fig.suptitle('Verified Qwen-2B prompt subset · BF16 / 4 s / stride 1 s\n95% video-bootstrap CI · full study still running')
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(out/('prompt_metrics.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
lines=['# Qwen Prompt 阶段性结果','',
       '仅覆盖已完整验收的 Qwen 2B P1/P2/P3／4秒条件；整体正式实验仍未完成。',
       '每条件 290 视频、37,188 窗口、1,111,808 帧；定位使用固定 140 视频。',
       'BF16、batch 4、无前缀缓存；配对视频 Bootstrap 1,000 次、seed 42。','',
       '|Prompt|Video Max AUROC|Video Max AP|Within Macro AUROC|Within Macro AP|Pooled AUROC|',
       '|---|---:|---:|---:|---:|---:|']
for name,report in reports.items():
    lines.append('|'+name.split('_')[2].upper()+'|'+'|'.join(f"{report['metrics'][k]['value']:.4%}" for k in ['video_max_auroc','video_max_ap','within_macro_auroc','within_macro_ap','pooled_auroc'])+'|')
lines+=['','配对差值以百分点表示，95% 视频 Bootstrap 区间未经多重比较校正。','',
        '|对比|Video Max AUROC 差值 [95% CI]|Within Macro AUROC 差值 [95% CI]|',
        '|---|---:|---:|']
for contrast,values in contrasts.items():
    cells=[]
    for key in ['video_max_auroc','within_macro_auroc']:
        value=values[key];ci=value['ci95']
        cells.append(f"{100*value['difference']:+.2f} [{100*ci['low']:+.2f}, {100*ci['high']:+.2f}]")
    lines.append('|'+contrast+'|'+'|'.join(cells)+'|')
lines+=['','其他置信区间、Mean/Top-k 敏感性及配对差值见 JSON。',
        'Prompt 和主实验 P1 已提前冻结；不根据这张表改变 Prompt 或训练停止规则。',
        '官方测试集已有历史探索记录，不称为独立盲测。',
        'Open-Jev Prompt 对照、多轮 MIL、规模对照及最终全矩阵分析仍待完成。','',
        '![阶段性指标](prompt_metrics.png)','']
(out/'README.md').write_text('\n'.join(lines))
print(json.dumps({name:{k:r['metrics'][k]['value'] for k in metrics} for name,r in reports.items()},indent=2))
