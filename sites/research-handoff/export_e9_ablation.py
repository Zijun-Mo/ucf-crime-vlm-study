"""Add the two frozen E9 ablations to the accepted handoff data."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / 'dist/data'
E9 = 'qwen_localization_v4_plateau_test'
ARMS = {'a': 'Qwen · A 打乱标签位置', 'b': 'Qwen · B 去掉 min–max'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_csv(path, rows, ids):
    with path.open(encoding='utf-8-sig', newline='') as src:
        reader = csv.reader(src)
        header = next(reader)
        old = [row for row in reader if row[0] not in ids]
    with path.open('w', encoding='utf-8-sig', newline='') as dest:
        writer = csv.writer(dest)
        writer.writerows([header, *old, *rows])


def main():
    experiment = ROOT / 'research/e9_ablation'
    final = read(experiment / 'final_results.json')
    protocol = read(experiment / 'protocol.json')
    assert final['status'] == 'complete' and final['active_assignments'] == 0
    records = read(DATA / 'test_videos.json')
    web = read(DATA / 'site.json')
    comparisons = read(DATA / 'comparisons.json')
    baseline = next(row for row in web['conditions'] if row['id'] == E9)
    assert baseline['metrics'] == final['conditions']['e9']
    ids = {f'qwen_e9_ablation_{arm}_test' for arm in ARMS}
    comparison_ids = {f'{name}_minus_e9' for name in ids}
    web['conditions'] = [row for row in web['conditions'] if row['id'] not in ids]
    comparisons['comparisons'] = [row for row in comparisons['comparisons'] if row['id'] not in comparison_ids]
    summary_rows, video_rows = [], []
    for arm, label in ARMS.items():
        name = f'qwen_e9_ablation_{arm}_test'
        source = ROOT / 'results' / name / 'summary.json'
        report = read(source)['reports']['candidate']
        evidence = ROOT / f'research/e9_ablation_{arm}_test'
        audit = read(evidence / 'complete_prediction_audit.json')
        selected = read(evidence / 'selected_checkpoint.json')
        training = read(evidence / 'training_acceptance.json')
        assert audit['passed'] and audit['optimizer_updates'] == 0
        assert (audit['videos'], audit['windows'], audit['frames']) == (290, 37188, 1111808)
        assert selected['selected_epoch'] == 9 and selected['optimizer_updates'] == 1413
        assert selected['checkpoint_tensor_sha256'] == audit['checkpoint_tensor_sha256']
        assert report['metrics'] == final['conditions'][arm]
        predictions = {}
        for record in records:
            video = record['name']
            path = source.parent / 'raw/predictions' / f'{video}.json'
            assert sha(path) == audit['prediction_sha256'][path.name]
            row = read(path)
            assert row['name'] == video and row['source_sha256'] == record['sha256']
            assert row['checkpoint_tensor_sha256'] == selected['checkpoint_tensor_sha256']
            predictions[video] = {k: row[k] for k in ('name', 'source_sha256', 'start_s', 'raw_logits')}
        configuration = {**baseline['configuration'],
                         'target_mode': protocol['arms'][arm], 'optimizer_updates': 1413,
                         'checkpoint_tensor_sha256': selected['checkpoint_tensor_sha256']}
        condition = {**baseline, 'id': name, 'label': label, 'method': 'ablation', 'cost': {},
                     'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
                     'metrics': report['metrics'], 'configuration': configuration, 'per_video': report['per_video']}
        web['conditions'].append(condition)
        comparisons['comparisons'].append(dict(id=f'{name}_minus_e9', coefficients={name: 1, E9: -1},
            metrics=final['contrasts_vs_e9'][arm], scope='290-video test, step 1413, paired video bootstrap'))
        write(DATA / 'predictions' / f'{name}.json', predictions)
        write(DATA / 'reports' / f'{name}.json', report)
        write(DATA / 'per_video' / f'{name}.json', report['per_video'])
        keys = ('epoch', 'dev_pooled_auroc', 'dev_pooled_ap', 'within_macro_auroc', 'video_max_ap', 'state_sha256')
        write(DATA / 'training' / f'{name}_development.json', {
            'training_updates': 1413, 'test_epoch': 9,
            'best_development_epoch': training['development_selected_epoch'],
            'validation': [{k: row[k] for k in keys} for row in training['validation']],
        })
        write(DATA / 'provenance' / f'{name}.json', {
            **{k: audit[k] for k in ('videos', 'windows', 'frames', 'optimizer_updates', 'checkpoint_file_sha256', 'checkpoint_tensor_sha256')},
            'training_updates': 1413, 'selected_epoch': 9, 'prediction_source_sha256': sha(source),
            'target_mode': protocol['arms'][arm], 'test_used_for_selection': False, 'gpu_released': True,
        })
        summary_keys = ('pooled_auroc', 'pooled_ap', 'video_max_auroc', 'video_max_ap', 'within_macro_auroc', 'within_macro_ap')
        summary_rows.append([name, label, 290, 1111808, *(report['metrics'][k]['value'] for k in summary_keys)])
        for row in report['per_video']:
            keys = ('name', 'category', 'frames', 'positive_frames', 'within_auroc', 'within_ap', 'max', 'mean', 'topk')
            video_rows.append([name, *(row[k] for k in keys)])
    assert len(web['conditions']) == 20
    web['comparisons'] = comparisons['comparisons']
    web['e9_ablation'] = {'auroc_decomposition': final['auroc_decomposition'], 'bootstrap': final['bootstrap']}
    write(DATA / 'analysis/e9_ablation.json', final)
    public_protocol = {k: protocol[k] for k in ('arms','epochs','updates','confirmatory_checkpoint','sampling_plan_sha256','matched_training_videos','active_anomaly_videos','bootstrap')}
    write(DATA / 'analysis/e9_ablation_protocol.json', public_protocol)
    write(DATA / 'comparisons.json', comparisons)
    write(DATA / 'summary.json', [{k: v for k, v in row.items() if k != 'per_video'} for row in web['conditions']])
    append_csv(DATA / 'summary.csv', summary_rows, ids)
    append_csv(DATA / 'per_video.csv', video_rows, ids)
    write(DATA / 'site.json', web)
    (DATA / 'site.js').write_text('window.RESEARCH_DATA=' + json.dumps(web, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + ';')


if __name__ == '__main__':
    main()
