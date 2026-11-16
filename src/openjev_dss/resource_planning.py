"""Select scale hardware from measured train-fixture probes, never test scores."""
import re
import math


def budgeted_runtime_minutes(balance_cu, rate_cu_hr, requested_minutes, reserve_cu=2):
    """Cap a watchdog using an observed rate; never plan against future grants."""
    values = (balance_cu, rate_cu_hr, requested_minutes, reserve_cu)
    if not all(math.isfinite(v) for v in values) or rate_cu_hr <= 0 or reserve_cu < 0:
        raise ValueError('A finite balance, positive observed rate and nonnegative reserve are required')
    affordable = math.floor(max(0, balance_cu - reserve_cu) * 60 / rate_cu_hr)
    minutes = min(math.floor(requested_minutes), affordable)
    if minutes < 15:
        raise ValueError('Less than 15 minutes affordable after the credit reserve; do not allocate')
    return minutes


def estimate_scale_cost(probe, receipt, model_config, upload_benchmark, windows=37188, videos=290):
    if probe.get('status') != 'passed' or probe.get('test_labels_used') is not False:
        raise ValueError('A successful, training-only probe is required')
    if (probe.get('dtype'), probe.get('batch'), probe.get('prefix_cache')) != ('bf16', 4, False):
        raise ValueError('Probe changed the controlled inference configuration')
    headroom = probe['gpu_gib'] - probe['peak_gpu_gib']
    if headroom < .25:
        raise ValueError('Less than 0.25 GiB measured VRAM headroom')
    match = re.search(r'Usage rate:\s*([0-9.]+)/hr', receipt.get('usage_allocated', ''))
    if not match or float(match.group(1)) <= 0:
        raise ValueError('No positive observed allocation rate')
    rate = float(match.group(1))
    throughput = probe['language_windows_s']
    vision_batch = min(probe['vision_batch4_s'])
    upload_benchmark = probe.get('upload_benchmark', upload_benchmark)
    trial = next(r for r in upload_benchmark['trials'] if r['mode'] == 'single_request' and r['exit_code'] == 0)
    bytes_s = upload_benchmark['source_bytes'] / trial['seconds']
    if min(throughput, vision_batch, bytes_s) <= 0:
        raise ValueError('Invalid measured timing')
    duration = probe['job']['duration_s']
    pairs = probe['pair_feature_gate']['allowed']
    units = windows + videos if pairs else windows * duration
    hidden = model_config['text_config']['hidden_size']
    feature_bytes = units * 49 * hidden * 2  # BF16; one independent video-tail pair.
    vision_batches = units / (4 * duration)
    parts = {'language_s': windows / throughput,
             'vision_s': vision_batches * vision_batch,
             'feature_upload_s': feature_bytes / bytes_s,
             'setup_allowance_s': probe['elapsed_s'] + 120}
    seconds = sum(parts.values())
    return {'gpu': receipt['gpu'], 'observed_rate_cu_hr': rate, 'vram_headroom_gib': headroom,
            'language_windows_s': throughput, 'pair_reuse_allowed': pairs,
            'feature_bytes_estimate': feature_bytes, 'seconds_estimate': seconds,
            'estimated_cu': seconds * rate / 3600, 'planning_cu_with_50pct_margin': seconds * rate / 2400,
            'components': parts,
            'limitations': 'Fixture-based estimate, not a guaranteed bound: extrapolated upload bandwidth and steady vision; excludes probe credits already incurred and source-video downloads.'}


def choose_scale_hardware(candidates, model_config, upload_benchmark):
    estimates, rejected = [], []
    for probe, receipt in candidates:
        try:
            row = estimate_scale_cost(probe, receipt, model_config, upload_benchmark)
            row['receipt'] = receipt.get('receipt_path')
            estimates.append(row)
        except (KeyError, ValueError, StopIteration) as error:
            rejected.append({'gpu': receipt.get('gpu'), 'reason': str(error), 'receipt': receipt.get('receipt_path')})
    if not estimates:
        raise ValueError('No measured scale candidate fits the unchanged full-test configuration')
    selected = min(estimates, key=lambda row: (row['estimated_cu'], row['seconds_estimate']))
    return {'selected_gpu': selected['gpu'], 'selection': 'lowest estimated CU among successful full-precision batch-4 training-only probes',
            'candidates': estimates, 'rejected': rejected, 'selected_estimate': selected,
            'test_scores_used': False}
