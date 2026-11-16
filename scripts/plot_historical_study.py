"""Publication-style figures for historical dual-level metrics; no model selection."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.study_evaluation import frame_labels, frame_scores

folder = ROOT / 'results/study_historical'
out = folder / 'figures'
out.mkdir(exist_ok=True)
names = ['qwen_zs', 'openjev_zs', 'openjev_mil', 'openjev_mil_dss']
labels = ['Qwen zero-shot', 'Open-Jev zero-shot', 'Open-Jev MIL', 'Open-Jev MIL+DSS']
colors = ['#4575b4', '#74add1', '#d73027', '#fdae61']
reports = {name: json.loads((folder / (name + '_s1.json')).read_text()) for name in names}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, metric, title in zip(axes, ['video_max_auroc', 'within_macro_auroc', 'pooled_auroc'],
                             ['Between-video AUROC', 'Within-video macro AUROC', 'Pooled frame AUROC']):
    for i, name in enumerate(names):
        row = reports[name]['metrics'][metric]
        value, ci = row['value'] * 100, row['ci95']
        ax.errorbar(value, 3 - i, xerr=[[value - ci['low'] * 100], [ci['high'] * 100 - value]],
                    fmt='o', color=colors[i], capsize=3)
    ax.set_yticks(range(4), labels[::-1] if ax is axes[0] else [''] * 4)
    ax.set_title(title)
    ax.set_xlabel('AUROC (%) / video-bootstrap 95% CI')
    ax.grid(axis='x', alpha=.2)
fig.suptitle('Historical evidence: zero-shot FP16 vs. MIL BF16 (conditions differ)', fontsize=12)
fig.tight_layout()
fig.savefig(out / 'dual_level_historical.png', dpi=180)
fig.savefig(out / 'dual_level_historical.svg')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, metric, title in zip(axes, ['video_max_auroc', 'within_macro_auroc', 'pooled_auroc'],
                             ['Between-video', 'Within-video', 'Pooled frames']):
    for name, label, color in zip(names, labels, colors):
        values = [json.loads((folder / f'{name}_s{s}.json').read_text())['metrics'][metric]['value'] * 100 for s in [1, 2, 4]]
        ax.plot([1, 2, 4], values, 'o-', label=label, color=color)
    ax.set_title(title)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel('Window stride (s); fixed duration 4 s')
    ax.set_ylabel('AUROC (%)')
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'stride_historical.png', dpi=180)
fig.savefig(out / 'stride_historical.svg')
plt.close(fig)

manifest = json.loads((ROOT / 'results/formal_stride1/manifest.json').read_text())['videos']
index = {r['name']: r for r in manifest}
mixed = [r for r in reports['openjev_mil']['per_video'] if r['within_auroc'] is not None]
# Descriptive selection disclosed in metadata, never used to choose prompts/models.
best = max(mixed, key=lambda r: (r['within_auroc'], r['name']))['name']
worst = min(mixed, key=lambda r: (r['within_auroc'], r['name']))['name']
fig, axes = plt.subplots(2, 1, figsize=(11, 5.4))
for ax, name, caption in zip(axes, [best, worst], ['highest', 'lowest']):
    record = index[name]
    y = frame_labels(record)
    paths = ['results/formal_stride1/native_qwen', 'results/formal_stride1/merged_openjev',
             'results/finetuned_test/A', 'results/finetuned_test/B']
    for path, label, color in zip(paths, labels, colors):
        p = json.loads((ROOT / path / (name + '.json')).read_text())
        key = 'raw_logits' if 'raw_logits' in p else 'scores'
        z = frame_scores(record, p['start_s'], p[key])
        # Normalize independently for display only; reported metrics use original logits.
        span = z.max() - z.min()
        z = (z - z.min()) / span if span else np.zeros_like(z)
        ax.plot(np.arange(len(z))[::30] / record['fps'], z[::30], color=color, label=label, lw=1)
    ax.fill_between(np.arange(len(y)) / record['fps'], 0, 1, where=y.astype(bool), color='#555555', alpha=.13)
    ax.set_title(f'{name}: {caption} MIL within-video AUROC (descriptive selection)')
    ax.set_ylabel('Display-normalized score')
    ax.set_xlabel('Time (s); gray = annotated anomaly')
axes[0].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(out / 'localization_cases_historical.png', dpi=180)
fig.savefig(out / 'localization_cases_historical.svg')
plt.close(fig)
(out / 'case_selection.json').write_text(json.dumps({'best': best, 'worst': worst,
    'selection': 'extrema of historical Open-Jev MIL within-video AUROC; descriptive only',
    'display': 'each curve independently min-max normalized; metrics retain original raw logits'}, indent=2) + '\n')

lines = ['# 历史双层评估（本地 CPU 分析）', '',
         '这些结果复用历史预测，不消耗 GPU credit；跨阶段 FP16/BF16 与缓存条件不同。', '',
         '|方案|Video Max AUROC|Within Macro AUROC|Pooled AUROC|', '|---|---:|---:|---:|']
for name, label in zip(names, labels):
    v = reports[name]['metrics']
    lines.append(f"|{label}|{v['video_max_auroc']['value']:.4%}|{v['within_macro_auroc']['value']:.4%}|{v['pooled_auroc']['value']:.4%}|")
lines += ['', '各组均包含 290 个视频、1,111,808 帧；视频内指标在同一 140 个混合标签视频上计算。',
          '以视频为单位配对 Bootstrap 1,000 次、seed=42；Max 是主聚合，Mean/Top-k 为预先固定的敏感性分析。',
          '详细 AP、置信区间、逐视频数值、步长对比和配对差值见同目录 JSON。', '',
          '![双层指标](figures/dual_level_historical.png)', '', '![步长](figures/stride_historical.png)', '',
          '![定位案例](figures/localization_cases_historical.png)', '',
          '以上仅完成历史证据分析；新的统一条件 GPU 实验仍由 `research/state.json` 跟踪。']
(folder / 'README.md').write_text('\n'.join(lines) + '\n')
print('Wrote three figure pairs and historical summary')
