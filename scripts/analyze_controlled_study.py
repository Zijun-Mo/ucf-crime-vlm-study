"""Validate complete controlled score bundles and calculate paired study contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.study_evaluation import evaluate, histogram_bootstrap, interval, resampling_weights
from openjev_dss.producer_audit import archived_producer_signature, audit_formal_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all', action='store_true', help='Fail unless every mandatory score bundle is present')
    parser.add_argument('--formal-v2', action='store_true', help='Use validation-selected multi-epoch MIL arms as the primary comparison')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'research/protocol.json').read_text())
    registry = json.loads((ROOT / 'research/inference_jobs.json').read_text())['groups']
    expected = [job for group in registry.values() for job in group]
    if args.formal_v2:
        expected = [j for j in expected if j['mode'] not in ['qwen_mil', 'openjev_mil']]
        expected += json.loads((ROOT / 'research/formal_inference_jobs.json').read_text())['groups']['formal_mil']
    required = [job['name'] for job in expected if not job.get('optional')]
    manifest_path = ROOT / 'results/formal_stride1/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol['test_manifest_sha256']
    records = json.loads(manifest_path.read_text())['videos']
    weights = resampling_weights(len(records), protocol['evaluation']['bootstrap'], protocol['evaluation']['seed'])
    out = ROOT / ('results/study_formal_v2' if args.formal_v2 else 'results/study_controlled')
    out.mkdir(exist_ok=True)
    reports, samples, missing = {}, {}, []
    canonical = {r['name']: r for r in json.loads((ROOT / 'results/formal_stride1/config.json').read_text())['canonical']}
    preflight_versions = json.loads((ROOT / 'research/gpu_preflight_result.json').read_text())['versions']
    receipts = [(p, json.loads(p.read_text())) for p in sorted((ROOT / 'research/runs').glob('*/receipt.json'))]
    producer_signatures = set()
    eligible = None
    for job in expected:
        directory = ROOT / 'results/controlled' / job['name']
        if not (directory / 'complete.json').exists():
            missing.append(job['name'])
            continue
        config = json.loads((directory / 'config.json').read_text())
        fingerprint = config.pop('fingerprint')
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == fingerprint
        assert all(config[key] == value for key, value in job.items())
        assert config['protocol_sha256'] == hashlib.sha256((ROOT / 'research/protocol.json').read_bytes()).hexdigest()
        assert (config['dtype'], config['batch'], config['prefix_cache']) == ('bf16', 4, False)
        producers = [(p, receipt) for p, receipt in receipts if job['name'] in receipt.get('completed', [])]
        assert producers, 'No controller receipt proves this condition was durably completed'
        producer_path, producer = producers[-1]
        producer_signatures.add(archived_producer_signature(ROOT,config,producer))
        assert len(producer_signatures) == 1, 'Unreviewed change to the shared numerical producer/environment'
        training_receipt = None
        if config.get('training_protocol_version') == 2:
            training_receipt = audit_formal_training(ROOT,config,receipts,json.loads((ROOT / 'research/training_protocol_v2.json').read_text()))
        environment = json.loads((producer_path.parent / 'environment.json').read_text())
        assert environment['versions'] == preflight_versions, 'Runtime versions differ from numerical preflight'
        assert config['torch'] == environment['versions']['torch']
        marker = json.loads((directory / 'complete.json').read_text())
        assert marker == {'complete': True, 'fingerprint': fingerprint, 'videos': 290, 'windows': 37188, 'frames': 1111808}
        predictions = {p.stem: json.loads(p.read_text()) for p in (directory / 'predictions').glob('*.json')}
        assert set(predictions) == {r['name'] for r in records}
        for record in records:
            p = predictions[record['name']]
            assert p['name'] == record['name'] and p['source_sha256'] == record['sha256']
            assert p['fingerprint'] == fingerprint
            assert p['canonical_sha256'] == canonical[record['name']]['canonical_sha256']
        report, bootstrap = evaluate(records, predictions, 'raw_logits', weights, duration=job['duration_s'])
        assert sum(row['windows'] for row in report['per_video']) == 37188
        if eligible is None:
            eligible = report['eligible_within_videos']
        assert eligible == report['eligible_within_videos'] and len(eligible) == 140
        report.update(job=job, config_fingerprint=fingerprint, training_receipt=training_receipt)
        # Category video metrics compare one anomaly category to the same 150 normal videos.
        categories = {}
        for category in sorted({r['category'] for r in records} - {'Normal'}):
            mask = np.array([r['category'] in ['Normal', category] for r in report['per_video']])
            selected = [r for r, take in zip(report['per_video'], mask) if take]
            y = [np.array([r['label']]) for r in selected]
            z = [np.array([r['max']]) for r in selected]
            point, boot = histogram_bootstrap(y, z, weights[:, mask])
            loc = np.array([r['category'] == category and r['within_auroc'] is not None for r in report['per_video']])
            values = np.array([r['within_auroc'] for r, take in zip(report['per_video'], loc) if take])
            wc = weights[:, loc]
            with np.errstate(invalid='ignore', divide='ignore'):
                macros = (wc @ values) / wc.sum(axis=1)
            categories[category] = {'anomaly_videos': sum(r['category'] == category for r in selected),
                'normal_comparison_videos': 150, 'video_max_auroc': float(point[0]), 'video_max_ap': float(point[1]),
                'video_auroc_ci95': interval(boot[:, 0]), 'within_macro_auroc': float(values.mean()),
                'within_auroc_ci95': interval(macros)}
        report['categories'] = categories
        reports[job['name']], samples[job['name']] = report, bootstrap
        (out / (job['name'] + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print('ANALYZED', job['name'], flush=True)
    comparisons = {}

    def contrast(name, coefficients):
        if not set(coefficients) <= set(reports):
            return
        comparisons[name] = {'coefficients': coefficients, 'metrics': {
            metric: {'difference': sum(c * reports[j]['metrics'][metric]['value'] for j, c in coefficients.items()),
                     'ci95': interval(sum(c * samples[j][metric] for j, c in coefficients.items()))}
            for metric in samples[next(iter(coefficients))]}}

    qwen_mil = 'qwen_formal_mil_p1_4s' if args.formal_v2 else 'qwen_mil_p1_4s'
    openjev_mil = 'openjev_formal_mil_p1_4s' if args.formal_v2 else 'openjev_mil_p1_4s'
    contrast('decision_adaptation_zero_shot', {'openjev_zs_p1_4s': 1, 'qwen_zs_p1_4s': -1})
    contrast('decision_adaptation_after_mil', {openjev_mil: 1, qwen_mil: -1})
    contrast('weak_supervision_qwen', {qwen_mil: 1, 'qwen_zs_p1_4s': -1})
    contrast('weak_supervision_openjev', {openjev_mil: 1, 'openjev_zs_p1_4s': -1})
    contrast('adaptation_by_weak_supervision_interaction', {openjev_mil: 1, qwen_mil: -1,
                                                          'openjev_zs_p1_4s': -1, 'qwen_zs_p1_4s': 1})
    for mode in ['qwen_zs', 'openjev_zs']:
        baseline = mode + '_p1_4s'
        for prompt in ['p2', 'p3']:
            contrast(mode + '_' + prompt + '_minus_p1', {mode + '_' + prompt + '_4s': 1, baseline: -1})
        for duration in [2, 8]:
            contrast(mode + f'_{duration}s_minus_4s', {mode + f'_p1_{duration}s': 1, baseline: -1})
    contrast('scale_9b_minus_2b', {'qwen_9b_p1_4s': 1, 'qwen_zs_p1_4s': -1})
    contrast('scale_27b_minus_9b', {'qwen_27b_p1_4s': 1, 'qwen_9b_p1_4s': -1})
    summary = {'complete_mandatory_predictions': set(required) <= set(reports), 'available': sorted(reports),
               'missing_mandatory': sorted(set(required) - set(reports)), 'missing_optional': sorted(set(missing) - set(required)),
               'training_protocol_version': 2 if args.formal_v2 else 1,
               'bootstrap_replicates': len(weights), 'bootstrap_seed': protocol['evaluation']['seed'],
               'paired_contrasts': comparisons,
               'note': 'Prediction analysis completeness does not prove training receipts, cost audit or final figures are complete.'}
    (out / 'comparisons.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'paired_contrasts'}, indent=2))
    if args.require_all and summary['missing_mandatory']:
        raise SystemExit('Mandatory full-test predictions still missing')


if __name__ == '__main__':
    main()
