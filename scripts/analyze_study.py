"""Analyze frozen historical predictions without consuming GPU credits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.study_evaluation import evaluate, interval, resampling_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/study_historical')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / 'research/protocol.json').read_text())
    manifest = ROOT / 'results/formal_stride1/manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == protocol['test_manifest_sha256']
    records = json.loads(manifest.read_text())['videos']
    cfg = protocol['evaluation']
    weights = resampling_weights(len(records), cfg['bootstrap'], cfg['seed'])
    np.save(args.out / 'bootstrap_video_multiplicities.npy', weights, allow_pickle=False)
    modes = {'qwen_zs': ('results/formal_stride1/native_qwen', 'scores'),
             'openjev_zs': ('results/formal_stride1/merged_openjev', 'scores'),
             'openjev_mil': ('results/finetuned_test/A', 'raw_logits'),
             'openjev_mil_dss': ('results/finetuned_test/B', 'raw_logits')}
    reports, boot = {}, {}
    eligible = None
    for mode, (folder, key) in modes.items():
        predictions = {p.stem: json.loads(p.read_text()) for p in (ROOT / folder).glob('*.json')}
        for stride in [1, 2, 4]:
            name = f'{mode}_s{stride}'
            report, samples = evaluate(records, predictions, key, weights, stride=stride)
            if eligible is None:
                eligible = report['eligible_within_videos']
            assert eligible == report['eligible_within_videos']
            reports[name], boot[name] = report, samples
            (args.out / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
            print(name, {k: round(v['value'], 6) for k, v in report['metrics'].items() if k in ['video_max_auroc', 'within_macro_auroc', 'pooled_auroc']}, flush=True)
    comparisons = {}
    pairs = [('openjev_zs_s1', 'qwen_zs_s1'), ('openjev_mil_dss_s1', 'openjev_mil_s1'),
             ('openjev_mil_s1', 'openjev_zs_s1')]
    for mode in modes:
        pairs += [(f'{mode}_s{stride}', f'{mode}_s1') for stride in [2, 4]]
    for a, b in pairs:
        comparisons[a + '_minus_' + b] = {
            'condition_note': 'cross-stage BF16/FP16 and cache confounding' if (a, b) == ('openjev_mil_s1', 'openjev_zs_s1') else 'within historical paired setup',
            'metrics': {k: {'difference': reports[a]['metrics'][k]['value'] - reports[b]['metrics'][k]['value'],
                            'ci95': interval(boot[a][k] - boot[b][k])} for k in boot[a]}}
    output = {'scope': 'historical evidence; not the new matched-condition experiment',
              'protocol_sha256': hashlib.sha256((ROOT / 'research/protocol.json').read_bytes()).hexdigest(),
              'eligible_within_videos': eligible, 'bootstrap_seed': cfg['seed'], 'bootstrap_replicates': cfg['bootstrap'],
              'paired_comparisons': comparisons}
    (args.out / 'comparisons.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
