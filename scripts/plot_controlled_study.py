"""Render verified controlled-study metrics, disclosed cases and observed compute costs."""
import argparse
import datetime
import json
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.study_evaluation import frame_labels, frame_scores

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, default=ROOT, help='Repository-shaped input export; defaults to this repository')
parser.add_argument('--formal-v2', action='store_true')
args = parser.parse_args()
ROOT = args.data_root.resolve()

folder = ROOT / ('results/study_formal_v2' if args.formal_v2 else 'results/study_controlled')
summary = json.loads((folder / 'comparisons.json').read_text())
reports = {name: json.loads((folder / (name + '.json')).read_text()) for name in summary['available']}
if not reports:
    raise SystemExit('No complete, analyzed controlled score bundle is available')
out = folder / 'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
qwen_mil = 'qwen_formal_mil_p1_4s' if args.formal_v2 else 'qwen_mil_p1_4s'
openjev_mil = 'openjev_formal_mil_p1_4s' if args.formal_v2 else 'openjev_mil_p1_4s'
core = ['qwen_zs_p1_4s', 'openjev_zs_p1_4s', qwen_mil, openjev_mil]
labels = dict(zip(core, ['Qwen zero-shot', 'Open-Jev zero-shot', 'Qwen MIL', 'Open-Jev MIL']))
colors = dict(zip(core, ['#4575b4', '#74add1', '#d73027', '#fdae61']))
metrics = ['video_max_auroc', 'within_macro_auroc', 'pooled_auroc']
titles = ['Between-video AUROC', 'Within-video macro AUROC', 'Pooled frame AUROC']
created = []
layout_only = (ROOT / 'LAYOUT_ONLY.txt').exists()

def label_layout(fig):
    if layout_only:
        fig.text(.5, .52, 'LAYOUT CHECK / SYNTHETIC CONDITIONS', ha='center', va='center',
                 rotation=20, fontsize=24, color='#b2182b', alpha=.45)


def save(fig, name):
    label_layout(fig)
    fig.tight_layout()
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out / (name + '.svg'), bbox_inches='tight')
    # Normalize SVG generator whitespace for clean text diffs.
    path = out / (name + '.svg')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    created.append(name)


available_core = [name for name in core if name in reports]
if available_core:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        for i, name in enumerate(available_core):
            row = reports[name]['metrics'][metric]
            value, ci = row['value'] * 100, row['ci95']
            ax.errorbar(value, len(available_core) - i - 1,
                        xerr=[[max(0, value - ci['low'] * 100)], [max(0, ci['high'] * 100 - value)]],
                        fmt='o', color=colors[name], capsize=3)
        ax.set_yticks(range(len(available_core)), [labels[n] for n in available_core][::-1] if ax is axes[0] else [''] * len(available_core))
        ax.set_title(title)
        ax.set_xlabel('AUROC (%) · 95% video-bootstrap CI')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Controlled inference: BF16, no prefix cache, P1, 4 seconds, stride 1 second')
    save(fig, 'core_dual_level')

for factor in ['prompt', 'duration']:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    plotted = False
    for mode, label, color in [('qwen_zs', 'Qwen zero-shot', '#4575b4'), ('openjev_zs', 'Open-Jev zero-shot', '#d73027')]:
        names = [mode + '_' + p + '_4s' for p in ['p1', 'p2', 'p3']] if factor == 'prompt' else [mode + f'_p1_{d}s' for d in [2, 4, 8]]
        if not all(name in reports for name in names):
            continue
        plotted = True
        x = np.arange(3) if factor == 'prompt' else np.array([2, 4, 8])
        for ax, metric, title in zip(axes, metrics, titles):
            y = np.array([reports[n]['metrics'][metric]['value'] * 100 for n in names])
            ax.plot(x, y, 'o-', color=color, label=label)
            low = np.array([reports[n]['metrics'][metric]['ci95']['low'] * 100 for n in names])
            high = np.array([reports[n]['metrics'][metric]['ci95']['high'] * 100 for n in names])
            ax.fill_between(x, low, high, color=color, alpha=.1)
            ax.set_xticks(x, ['P1', 'P2', 'P3'] if factor == 'prompt' else ['2', '4', '8'])
            ax.set_xlabel('Prompt (fixed 4 s)' if factor == 'prompt' else 'Duration (s); fixed 2 fps / stride 1 s')
            ax.set_ylabel('AUROC (%)')
            ax.set_title(title)
            ax.grid(alpha=.2)
    if plotted:
        axes[0].legend(fontsize=8)
        save(fig, factor + '_sensitivity')
    else:
        plt.close(fig)

if available_core:
    categories = sorted(reports[available_core[0]]['categories'])
    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .045], left=.13, right=.94,
                          bottom=.24, top=.93, wspace=.6)
    axes = [fig.add_subplot(grid[0, i]) for i in [0, 1]]
    color_axis = fig.add_subplot(grid[0, 2])
    for ax, metric, title in zip(axes, ['video_max_auroc', 'within_macro_auroc'], ['Category vs. 150 normal videos', 'Within-category macro localization']):
        data = np.array([[reports[n]['categories'][cat][metric] * 100 for n in available_core] for cat in categories])
        im = ax.imshow(data, aspect='auto', vmin=0, vmax=100, cmap='viridis')
        ax.set_yticks(range(len(categories)), categories)
        ax.set_xticks(range(len(available_core)), [labels[n] for n in available_core], rotation=28, ha='right')
        ax.set_title(title)
        for i in range(len(categories)):
            for j in range(len(available_core)):
                ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if data[i,j] < 55 else 'black')
    fig.colorbar(im, cax=color_axis, label='AUROC (%)')
    label_layout(fig)
    fig.savefig(out / 'category_dual_level.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'category_dual_level.svg', bbox_inches='tight')
    plt.close(fig)
    created.append('category_dual_level')

reference = openjev_mil
if reference in reports:
    manifest = json.loads((ROOT / 'results/formal_stride1/manifest.json').read_text())['videos']
    index = {r['name']: r for r in manifest}
    mixed = [r for r in reports[reference]['per_video'] if r['within_auroc'] is not None]
    best = max(mixed, key=lambda r: (r['within_auroc'], r['name']))['name']
    worst = min(mixed, key=lambda r: (r['within_auroc'], r['name']))['name']
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4))
    for ax, name, caption in zip(axes, [best, worst], ['highest', 'lowest']):
        record = index[name]
        y = frame_labels(record)
        for method in available_core:
            p = json.loads((ROOT / 'results/controlled' / method / 'predictions' / (name + '.json')).read_text())
            z = frame_scores(record, p['start_s'], p['raw_logits'], duration=4)
            z = (z - z.min()) / (z.max() - z.min()) if z.max() != z.min() else np.zeros_like(z)
            ax.plot(np.arange(len(z))[::30] / record['fps'], z[::30], label=labels[method], color=colors[method], lw=1)
        ax.fill_between(np.arange(len(y)) / record['fps'], 0, 1, where=y.astype(bool), color='#555555', alpha=.13)
        ax.set_title(f'{name}: {caption} Open-Jev MIL within-video AUROC (descriptive selection)')
        ax.set_xlabel('Time (s); gray = annotated anomaly')
        ax.set_ylabel('Display-normalized score')
    axes[0].legend(fontsize=8, ncol=2)
    save(fig, 'localization_cases')
    (out / 'case_selection.json').write_text(json.dumps({'best': best, 'worst': worst, 'reference': reference,
        'selection': 'complete test extrema; descriptive only, never used for model/prompt selection',
        'display': 'independent min-max normalization; numerical metrics retain original raw logits'}, indent=2) + '\n')

# CU estimates use each actually observed rate. Balances may lag; never label estimates as debited CU.
receipts = [json.loads(p.read_text()) for p in sorted((ROOT / 'research/runs').glob('*/receipt.json'))]
rate_by_job = {}
prefetch_by_job = {}
group_costs = []
for receipt in receipts:
    match = re.search(r'Usage rate:\s*([0-9.]+)/hr', receipt.get('usage_allocated', ''))
    if not match:
        continue
    rate = float(match.group(1))
    for name in receipt.get('completed', []):
        rate_by_job[name] = rate
        if receipt.get('feature_prefetch_receipt'):prefetch_by_job[name]=receipt['feature_prefetch_receipt']
    start = receipt.get('allocated_utc', receipt.get('started_utc'))
    finish = receipt.get('release_observed_utc', receipt.get('finished_utc'))
    if start and finish:
        def timestamp(text):
            return datetime.datetime.strptime(text, '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc) if 'T' in text and '-' not in text else datetime.datetime.fromisoformat(text)
        seconds = max(0, (timestamp(finish) - timestamp(start)).total_seconds())
        group_costs.append({'job': receipt['job'], 'status': receipt['status'], 'gpu': receipt['gpu'],
                            'observed_rate_cu_hr': rate, 'runtime_bound_seconds': seconds,
                            'estimated_cu': rate * seconds / 3600,
                            'timing_note': 'allocation/release observation bound; includes setup and failed attempts',
                            'receipt': next((str(p.relative_to(ROOT)) for p in (ROOT / 'research/runs').glob('*/receipt.json') if json.loads(p.read_text()) == receipt), None)})
condition_costs = {}
for name in reports:
    status = json.loads((ROOT / 'results/controlled' / name / 'status.json').read_text())
    # A resumed bundle's status counts all videos but times only the new segment.
    # Add earlier incomplete segments with the SAME producer fingerprint.
    segments = [status]
    segment_receipts = []
    for receipt_path in sorted((ROOT / 'research/runs').glob('*/receipt.json')):
        result_path = receipt_path.parent / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        earlier = result.get('active_status', {})
        if (earlier.get('job') == name and earlier.get('status') != 'complete' and
                earlier.get('fingerprint') == reports[name]['config_fingerprint']):
            segments.append(earlier)
            segment_receipts.append(str(receipt_path.relative_to(ROOT)))
    seconds = sum(r['elapsed_s'] for r in segments)
    language_s = sum(r['language_s'] for r in segments)
    vision_s = sum(r['vision_or_cache_s'] for r in segments)
    condition_costs[name] = {'elapsed_s': seconds, 'language_s': language_s,
        'vision_or_cache_s': vision_s, 'end_to_end_windows_s': 37188 / seconds,
        'language_windows_s': 37188 / language_s, 'peak_gpu_gib': status['peak_gpu_gib'],
        'earlier_incomplete_segments': segment_receipts,
        'peak_note': 'Completed segment peak; aborted segments may not record a peak',
        'gpu': status['gpu'], 'observed_rate_cu_hr': rate_by_job.get(name),
        'estimated_phase_cu': seconds * rate_by_job[name] / 3600 if name in rate_by_job else None,
        'frontloaded_cache_preparation_receipt':prefetch_by_job.get(name),
        'timing_note':'Per-condition times exclude shared staging and final background-cache drain; allocation/group totals include them. Optional feature uploads may overlap computation in resumed segments; language-only timings are recorded separately.',
        'note': 'phase wall time includes I/O and matching earlier incomplete segments; separate group totals include setup/failures'}
(folder / 'compute_costs.json').write_text(json.dumps({'conditions': condition_costs, 'groups': group_costs,
    'estimated_total_group_cu': sum(r['estimated_cu'] for r in group_costs),
    'cu_note': 'estimates from observed rate and wall time; not an exact credit debit',
    'scope': 'full study group receipts; separate preflight sessions excluded here'}, indent=2) + '\n')
scale = [n for n in ['qwen_zs_p1_4s', 'qwen_9b_p1_4s', 'qwen_27b_p1_4s'] if n in reports]
if len(scale) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    x = np.arange(len(scale))
    for ax, metric, title in zip(axes, metrics, titles):
        y = np.array([reports[n]['metrics'][metric]['value'] * 100 for n in scale])
        low = np.array([reports[n]['metrics'][metric]['ci95']['low'] * 100 for n in scale])
        high = np.array([reports[n]['metrics'][metric]['ci95']['high'] * 100 for n in scale])
        ax.errorbar(x, y, yerr=[np.maximum(0, y-low), np.maximum(0, high-y)], fmt='o-', color='#4575b4', capsize=3)
        ax.set_xticks(x, [reports[n]['job']['scale'] for n in scale])
        ax.set_xlabel('Qwen scale; P1 / 4 s / stride 1 s')
        ax.set_ylabel('AUROC (%)')
        ax.set_title(title)
        ax.grid(alpha=.2)
    save(fig, 'scale_dual_level')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, name in enumerate(scale):
        label = reports[name]['job']['scale']
        cost = condition_costs[name]
        axes[0].scatter(cost['end_to_end_windows_s'], reports[name]['metrics']['video_max_auroc']['value'] * 100, s=50)
        axes[0].annotate(label, (cost['end_to_end_windows_s'], reports[name]['metrics']['video_max_auroc']['value'] * 100), xytext=(5, 5), textcoords='offset points')
        axes[1].bar(label, cost['peak_gpu_gib'])
        axes[2].bar(label, cost['estimated_phase_cu'] if cost['estimated_phase_cu'] is not None else np.nan)
        if cost['estimated_phase_cu'] is None:
            axes[2].text(i, 0, 'not recorded', ha='center', fontsize=8)
    axes[0].set_xlabel('End-to-end windows/s (including I/O)')
    axes[0].set_ylabel('Video Max AUROC (%)')
    axes[1].set_ylabel('Peak allocated VRAM (GiB)')
    axes[2].set_ylabel('Estimated inference-phase CU')
    save(fig, 'scale_cost_accuracy')

lines = ['# 统一条件实验结果', '',
         '所有已列入结果均通过完整测试清单校验：290 个视频、37,188 个窗口、1,111,808 帧。',
         'BF16、batch 4、无前缀缓存；定位指标使用同一 140 个混合标签视频。', '',
         '|实验|Video Max AUROC|Video Max AP|Within Macro AUROC|Within Macro AP|Pooled AUROC|Pooled AP|',
         '|---|---:|---:|---:|---:|---:|---:|']
for name, report in reports.items():
    m = report['metrics']
    lines.append('|' + name + '|' + '|'.join(f"{m[k]['value']:.4%}" for k in ['video_max_auroc', 'video_max_ap', 'within_macro_auroc', 'within_macro_ap', 'pooled_auroc', 'pooled_ap']) + '|')
lines += ['', '指标置信区间与配对差值见 JSON（视频簇 Bootstrap 1,000 次，seed 42）。',
          '模型和 Prompt 均预先固定；测试集已有历史探索记录，不称为独立盲测。',
          '成本区分观测费率推算与实际余额，包含 I/O 的吞吐不能解释为纯 GPU 计算速度。', '',
          '尚缺必需预测：' + (', '.join(summary['missing_mandatory']) or '无') + '。',
          '训练完成凭据、可选 27B 的资源审计及总验收另见 research/；本表不单独证明全任务完成。', '']
lines += ['## 配对因素差值', '',
          '差值方向由对照名称及 comparisons.json 的系数固定；下表单位为百分点。',
          '区间为以视频为单位的配对 Bootstrap 95% CI，未校正多个对照带来的多重检验。', '',
          '|对照|Video Max AUROC 差值 [95% CI]|Within Macro AUROC 差值 [95% CI]|',
          '|---|---:|---:|']
for name, contrast in summary['paired_contrasts'].items():
    cells=[]
    for metric in ['video_max_auroc','within_macro_auroc']:
        row=contrast['metrics'][metric];ci=row['ci95']
        cells.append(f"{row['difference']*100:+.3f} [{ci['low']*100:+.3f}, {ci['high']*100:+.3f}]")
    lines.append('|' + name + '|' + '|'.join(cells) + '|')
lines += ['', '## 视频聚合敏感性', '',
          'Max 为预先固定的主聚合；Mean 与 Top-k 仅作敏感性分析，不逐模型择优。', '',
          '|实验|Max AUROC|Mean AUROC|Top-k AUROC|Max AP|Mean AP|Top-k AP|',
          '|---|---:|---:|---:|---:|---:|---:|']
for name,report in reports.items():
    metrics=report['metrics']
    keys=['video_'+mode+'_'+measure for measure in ['auroc','ap'] for mode in ['max','mean','topk']]
    lines.append('|' + name + '|' + '|'.join(f"{metrics[key]['value']:.4%}" for key in keys) + '|')
lines += ['', '## 解释边界', '',
          'Qwen 使用原生 Yes/No token-logit 读出，Open-Jev 使用已发布的决策适配器与标量头；完整方案差值不能仅归因于 LoRA。',
          ('MIL 检查点只按完整验证集的固定 AP 规则选择；实际轮次、梯度检查和平台期证据见 ../formal_training_v2/。' if args.formal_v2 else '本表的旧单轮 MIL 结果不证明收敛；正式多轮训练证据另列。'),
          '训练只有一个固定随机种子；视频 Bootstrap 不能度量重新训练的种子差异。',
          '规模比较限于同系列模型，仍可能受到架构与预训练差异影响。测试集已有探索历史；未声明独立外部泛化验证。',
          '定位案例按完整测试集的最好/最差结果作描述性选择，显示曲线各自归一化；指标保留原始分数。',
          '开发集没有本轮事件阈值选择所需的时间区间标注，因此不补充事件级阈值指标。', '']
for name in created:
    lines += [f'![{name}](figures/{name}.png)', '']
(folder / 'README.md').write_text('\n'.join(lines))
print('Wrote controlled figures:', ', '.join(created))
