"""Build a portable, frozen research handoff from accepted result artifacts.

Never allocates compute, reruns a model, or changes original project artifacts.
"""
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIST = HERE / 'dist'
DATA = DIST / 'data'
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.study_evaluation import frame_labels, frame_scores
from sklearn.metrics import roc_auc_score, average_precision_score


def read(p):
    return json.loads((ROOT / p).read_text())


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    # Local reviewer paths are provenance metadata, not scientific observations.
    text = re.sub(r'/home/[^/]+/[^"\\]*?/(?:hydrogen_remote_videos|hydrogen_native_previews)/([^"\\]+)', r'review_evidence/\1', text)
    p.write_text(text)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, dst)
    if dst.suffix == ".md":
        text = dst.read_text()
        text = re.sub(r"\[Drive 结果目录\]\(https://drive\.google\.com/[^)]+\)", "原始私人云盘目录不随公开快照分发；结果见本资料包。", text)
        dst.write_text(text)


LABELS = {
    'qwen_zs_p1_4s': 'Qwen · 零样本 P1',
    'openjev_zs_p1_4s': 'Open-Jev · 零样本 P1',
    'qwen_formal_mil_p1_4s': 'Qwen · 旧 MIL',
    'openjev_formal_mil_p1_4s': 'Open-Jev · MIL',
    'qwen_storyboard_context_formal_v1': 'Qwen · 零样本＋拼图',
    'qwen_localization_v2_epoch1_test': 'Qwen · 教师约束 E1',
    'qwen_localization_v4_epoch2_test': 'Qwen · 软标签 E2',
    'qwen_localization_v4_epoch5_test': 'Qwen · 软标签 E5',
    'qwen_localization_v4_plateau_test': 'Qwen · 软标签 E9（最终）',
    'qwen_9b_p1_4s': 'Qwen 9B · 零样本 P1',
}
METRICS = ['pooled_auroc', 'pooled_ap', 'video_max_auroc', 'video_max_ap', 'within_macro_auroc', 'within_macro_ap']
SCORE_JS_IDS = {'qwen_zs_p1_4s', 'qwen_formal_mil_p1_4s', 'qwen_storyboard_context_formal_v1',
                'qwen_localization_v2_epoch1_test', 'qwen_localization_v4_epoch2_test',
                'qwen_localization_v4_epoch5_test', 'qwen_localization_v4_plateau_test'}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    inventory = read('results/future_work_v1/inventory.json')
    policy = read('research/localization_v4_plateau_test/policy.json')
    records = [{k: r[k] for k in ['name', 'category', 'label', 'fps', 'n_frames', 'intervals', 'sha256']} for r in policy['records']]
    assert len(records) == 290 and sum(r['n_frames'] for r in records) == 1111808
    write(DATA / 'test_videos.json', records)
    expected_names = {r['name'] for r in records}
    conditions = []
    source_manifest = {}
    all_predictions = {}
    costs = read('results/study_formal_v2/compute_costs.json')['conditions']

    for condition in inventory['conditions']:
        name = condition['name']
        source = condition['source']
        source_manifest[source] = digest(ROOT / source)
        original = read(source)
        is_core = source.startswith('results/study_formal_v2/')
        if is_core:
            report = original
            folder = ROOT / 'results/controlled' / name / 'predictions'
            key = 'raw_logits'
            config = read(f'results/controlled/{name}/config.json')
        elif name == 'qwen_storyboard_context_formal_v1':
            report = read('results/storyboard_context_formal_v1/result.json')['reports']['storyboard']
            folder = ROOT / 'results/storyboard_context_formal_v1/predictions'
            key = 'storyboard'
            config = dict(model_id=policy['model_id'], prompt='P1', duration_s=4, stride_s=1, dtype='bf16', prefix_cache=False,
                          training=False, overview_max_frames=8, overview_max_pages=2, image_token_budget=768)
        else:
            report = original['reports']['candidate']
            folder = ROOT / 'results' / name / 'raw/predictions'
            key = 'raw_logits'
            config = dict(model_id=policy['model_id'], prompt='P1', duration_s=4, stride_s=1, dtype='bf16', prefix_cache=False,
                          selected_epoch=original['selected_epoch'], checkpoint_tensor_sha256=original['checkpoint_tensor_sha256'])
        predictions = {}
        raw_hashes = {}
        for r in records:
            path = folder / (r['name'] + '.json')
            p = json.loads(path.read_text())
            assert p['source_sha256'] == r['sha256']
            assert len(p['start_s']) == len(p[key]) and all(math.isfinite(v) for v in p[key])
            assert p['start_s'] == list(range(math.ceil(r['n_frames'] / r['fps'])))
            predictions[r['name']] = {k: p[k] for k in ['name', 'source_sha256', 'fingerprint', 'start_s'] if k in p}
            predictions[r['name']]['raw_logits'] = p[key]
            for k in ['canonical_sha256', 'checkpoint_tensor_sha256', 'selected_epoch']:
                if k in p:
                    predictions[r['name']][k] = p[k]
            raw_hashes[str(path.relative_to(ROOT))] = digest(path)
        assert set(predictions) == expected_names and sum(len(p['raw_logits']) for p in predictions.values()) == 37188
        duration = config.get('duration_s', 4)
        metrics = report['metrics']
        assert all(abs(metrics[k]['value'] - condition['metrics'][k]) < 1e-12 for k in condition['metrics'])
        rows = report['per_video']
        assert len(rows) == 290
        write(DATA / 'predictions' / (name + '.json'), predictions)
        # Dynamic scripts allow the same interactive score viewer to work from file://.
        script = DATA / 'predictions' / (name + '.js')
        if name in SCORE_JS_IDS:
            script.write_text('window.RESEARCH_SCORES=window.RESEARCH_SCORES||{};window.RESEARCH_SCORES[' + json.dumps(name) + ']=' + json.dumps(predictions, separators=(',', ':')) + ';')
        elif script.exists():
            script.unlink()  # Remove only redundant browser wrappers generated by this exporter.
        write(DATA / 'reports' / (name + '.json'), report)
        write(DATA / 'per_video' / (name + '.json'), rows)
        write(DATA / 'provenance' / (name + '.json'), dict(source=source, source_sha256=source_manifest[source], original_prediction_sha256=raw_hashes, configuration=config))
        if name not in LABELS:
            LABELS[name] = ('Open-Jev' if name.startswith('openjev') else 'Qwen') + ' · 零样本 ' + config['prompt'] + f' · {duration}s'
        method = 'soft' if 'localization_v4' in name else 'teacher' if 'localization_v2' in name else 'storyboard' if 'storyboard' in name else 'mil' if 'mil' in name else 'zero'
        conditions.append(dict(id=name, label=LABELS[name], model='Open-Jev' if name.startswith('openjev') else 'Qwen',
                               scale='9B' if '9b' in name else '2B', method=method, prompt=config['prompt'], duration=duration,
                               metrics=metrics, cost=costs.get(name), source=source, source_sha256=source_manifest[source],
                               configuration=config, per_video=rows, windows=37188, videos=290, frames=1111808))
        all_predictions[name] = predictions
    assert len(conditions) == 18

    comparisons = []
    core_contrasts = read('results/study_formal_v2/comparisons.json')['paired_contrasts']
    for name, c in core_contrasts.items():
        comparisons.append(dict(id=name, coefficients=c['coefficients'], metrics=c['metrics'], scope='formal_paired_comparison'))
    for condition in conditions:
        name = condition['id']
        if condition['method'] not in ['soft', 'teacher', 'storyboard']:
            continue
        d = read(condition['source'])
        contrasts = d['paired_contrasts']
        if condition['method'] == 'storyboard':
            contrasts = contrasts['storyboard_minus_baseline']
        comparisons.append(dict(id=name + '_minus_zero', coefficients={name: 1, 'qwen_zs_p1_4s': -1}, metrics=contrasts, scope='paired_vs_zero'))
    direct = read('results/qwen_localization_v4_plateau_test/method_comparisons.json')
    write(DATA / 'comparisons.json', dict(comparisons=comparisons, final_model_comparisons=direct))
    for c in conditions:
        if c['method'] in ['soft', 'teacher', 'storyboard']:
            write(DATA / 'statistics' / (c['id'] + '.json'), read(c['source']))
    write(DATA / 'statistics/core_comparisons.json', read('results/study_formal_v2/comparisons.json'))
    write(DATA / 'statistics/final_comparisons.json', direct)

    history = []
    dev_sources = list((ROOT / 'results/localization_v4_dev').glob('epoch_*_comparison.json')) + list((ROOT / 'results/localization_v4_continuation_dev').glob('epoch_*_comparison.json')) + list((ROOT / 'results/localization_v4_plateau_dev').glob('epoch_*_comparison.json'))
    for p in sorted(dev_sources, key=lambda p: int(p.name.split('_')[1])):
        d = json.loads(p.read_text())
        history.append({k: d[k] for k in ['epoch', 'baseline', 'trained', 'guards', 'qualified_candidate', 'video_validation_count', 'frame_dev_count', 'within_video_count'] if k in d})
        write(DATA / 'training' / p.name, d)
    assert [h['epoch'] for h in history] == list(range(1, 10))
    write(DATA / 'training/history.json', history)

    # Scientific configurations only: omit operational paths, billing account data and process identifiers.
    for phase in ['localization_v2', 'localization_v3', 'localization_v4']:
        p = read(f'research/{phase}/policy.json')
        scientific = {k: p[k] for k in ['version', 'name', 'initialization', 'backbone_revision', 'input', 'teacher', 'selection', 'loss', 'optimizer', 'training', 'validation', 'sampling', 'notes'] if k in p}
        if 'validation' in scientific:
            scientific['validation'] = {k: v for k, v in scientific['validation'].items() if k not in ['annotations', 'quality_audit']}
        write(DATA / 'training' / (phase + '_scientific_policy.json'), scientific)
    protocol = read('research/protocol.json')
    write(DATA / 'protocol.json', {k: protocol[k] for k in ['version', 'frozen_at_utc', 'prompts', 'evaluation']})
    write(DATA / 'training/soft_sampling.py.json', {'source': 'src/openjev_dss/localization_soft.py', 'sha256': digest(ROOT / 'src/openjev_dss/localization_soft.py')})
    copy('src/openjev_dss/localization_soft.py', DATA / 'training/localization_soft.py')
    copy('research/localization_dev_v2/dev_annotations.json', DATA / 'training/dev_annotations.json')
    manifest = read('.colab-work/localization_manifest.json')
    train_records = [{k: r[k] for k in ['name', 'category', 'label', 'split', 'sha256']} for r in manifest['records'] if r['split'] in ['train', 'val', 'calibration']]
    write(DATA / 'training/splits.json', train_records)
    # The cached native teacher is an inference result, not a model or feature cache.
    teacher_folder = HERE / 'work/teacher'
    teachers = sorted(teacher_folder.glob('*.json'))
    assert len(teachers) == 1419, f'Complete teacher results required, have {len(teachers)}'
    relevant = {r['name']: r for r in train_records if r['split'] in ['train', 'val']}
    assert len(relevant) == 1419
    teacher_meta = {}
    for p in teachers:
        d = json.loads(p.read_text());r = relevant[d['name']]
        assert d['source_sha256'] == r['sha256'] and len(d['logits']) == d['windows']
        assert all(math.isfinite(v) for v in d['logits'])
        copy(p, DATA / 'training/teacher' / p.name)
        meta_path = ROOT / '.colab-work/formal_metadata' / p.name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            teacher_meta[d['name']] = {k: meta[k] for k in ['name', 'fps', 'n_frames', 'canonical_sha256', 'window_count'] if k in meta}
    write(DATA / 'training/teacher_metadata.json', teacher_meta)
    for epoch in range(1, 10):
        if epoch <= 2:
            root = ROOT / f'research/localization_v4/epoch_{epoch}_review'
        elif epoch <= 5:
            root = ROOT / f'research/localization_v4_continuation/epoch_{epoch}_review'
        else:
            root = ROOT / f'research/localization_v4_plateau/epoch_{epoch}_review'
        predictions = root / 'validation'
        if not predictions.exists():
            predictions = ROOT / f'results/localization_v4_dev/epoch_{epoch}'
        files = sorted(predictions.rglob('*.mp4.json'))
        if len(files) == 163:
            for p in files:
                copy(p, DATA / f'training/development_predictions/epoch_{epoch}' / p.name)
    selected = read('research/localization_v4_plateau/selected_checkpoint.json')
    write(DATA / 'training/selected_checkpoint.json', selected)
    write(DATA / 'training/selection_replay.json', read('research/localization_v4_plateau/stopping_rule_replay.json'))
    write(DATA / 'training/hard_candidate_preflight.json', read('research/localization_v3/cpu_preflight.json'))
    write(DATA / 'training/old_mil_training.json', read('results/formal_training_v2/summary.json'))

    analysis = read('results/future_work_v1/frame_transfer_analysis.json')
    per_video_transfer = read('results/future_work_v1/per_video_transfer.json')
    contributions = read('results/future_work_v1/auc_contributions.json')
    write(DATA / 'analysis/frame_transfer.json', analysis)
    write(DATA / 'analysis/per_video_transfer.json', per_video_transfer)
    write(DATA / 'analysis/auc_contributions.json', contributions)
    cases = read('results/qwen_localization_v4_plateau_test/case_selection.json')
    write(DATA / 'analysis/cases.json', cases)
    copy('results/qwen_localization_v4_plateau_test/case_frames/manifest.json', DATA / 'analysis/case_images_manifest.json')
    for p in (ROOT / 'results/qwen_localization_v4_plateau_test/case_frames').glob('*.jpg'):
        copy(p, DIST / 'assets/cases' / p.name)
    figure_sources = [ROOT / 'results/study_formal_v2/figures', ROOT / 'results/future_work_v1', ROOT / 'results/qwen_localization_v4_plateau_test', ROOT / 'results/formal_training_v2']
    for src in figure_sources:
        for p in src.glob('*'):
            if p.suffix in ['.png', '.svg']:
                copy(p, DIST / 'assets/figures' / src.name / p.name)

    # Historical outputs remain explicitly separate from the controlled study.
    for src in ['results/formal_stride1', 'results/finetuned_test']:
        for p in (ROOT / src).rglob('*'):
            if p.is_file() and p.suffix in ['.json', '.md', '.csv', '.png', '.svg', '.txt']:
                copy(p, DATA / 'history' / Path(src).name / p.relative_to(ROOT / src))
    history_folder = DATA / 'history'
    history_folder.mkdir(exist_ok=True)
    (history_folder / 'README.md').write_text('历史结果使用早期精度／缓存／训练协议，不与当前18条件混合计算。它们用于追溯研究过程；当前正式主表以 data/reports 为准。\n')
    write(DATA / 'provenance/original_report_sha256.json', source_manifest)
    acceptance = read('research/future_work_v1/final_acceptance.json')
    public_acceptance = {k: acceptance[k] for k in ['status', 'verified_utc', 'total_formal_conditions', 'videos', 'windows', 'frames', 'selected_epoch', 'selected_optimizer_updates', 'unvalidated_tail_updates', 'checkpoint_file_sha256', 'checkpoint_tensor_sha256', 'all_prediction_hashes_verified', 'test_used_for_selection', 'formal_test_optimizer_updates', 'localization_noninferiority_proven', 'gpu_released']}
    write(DATA / 'provenance/acceptance.json', public_acceptance)
    write(DATA / 'provenance/budget.json', acceptance['budget'])
    for src in ['research/future_work_v1/aigc_usage.json', 'research/localization_v4/ai_use_and_review.json', 'research/localization_v4_plateau_test/ai_use_and_review.json']:
        write(DATA / 'provenance' / (Path(src).parent.name + '_ai_review.json'), read(src))
    write(DATA / 'course.json', {k: read('research/future_work_v1/course_contract.json')[k] for k in ['scope', 'report_body_pages', 'requires', 'deadlines', 'deadline_timezone', 'new_algorithm_or_sota_required']})

    # Flat CSVs are convenient for reports and spreadsheet software; numeric values stay [0,1].
    stream = io.StringIO();writer = csv.writer(stream)
    writer.writerow(['condition', 'label', 'videos', 'frames'] + METRICS)
    for c in conditions:
        writer.writerow([c['id'], c['label'], 290, 1111808] + [c['metrics'][m]['value'] for m in METRICS])
    (DATA / 'summary.csv').write_text('\ufeff' + stream.getvalue())
    write(DATA / 'summary.json', [{k: v for k, v in c.items() if k != 'per_video'} for c in conditions])
    stream = io.StringIO();writer = csv.writer(stream)
    writer.writerow(['condition', 'video', 'category', 'frames', 'positive_frames', 'within_auroc', 'within_ap', 'video_max_logit', 'video_mean_logit', 'video_topk_logit'])
    for c in conditions:
        for r in c['per_video']:
            writer.writerow([c['id'], r['name'], r['category'], r['frames'], r['positive_frames'], r['within_auroc'], r['within_ap'], r['max'], r['mean'], r['topk']])
    (DATA / 'per_video.csv').write_text('\ufeff' + stream.getvalue())
    web = dict(snapshot='2026-11-15 · frozen', videos=records, conditions=conditions, comparisons=comparisons,
               final_comparisons=direct, prompts=protocol['prompts'], training_history=history, analysis=analysis,
               contributions=contributions, per_video_transfer=per_video_transfer, cases=cases,
               selected_checkpoint=selected, budget=acceptance['budget'], course=read('research/future_work_v1/course_contract.json')['deadlines'])
    write(DATA / 'site.json', web)
    (DATA / 'site.js').write_text('window.RESEARCH_DATA=' + json.dumps(web, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';')
    print(json.dumps({'conditions': len(conditions), 'predictions': len(conditions)*290, 'teacher_results': len(teachers), 'development_epochs': len(history), 'site_data_bytes': (DATA/'site.js').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
