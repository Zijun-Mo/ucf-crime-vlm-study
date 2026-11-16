"""Evaluate both validation-selected formal MIL checkpoints using the frozen inference core.

Set UCF_GROUP and UCF_DEADLINE in the kernel environment. Source Drive mount is
read-only; new features/results are uploaded under remaining_20261108_v1 only.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

work = Path('/content/ucf_project')
sys.path.insert(0, '/content/ucf_deps')
sys.path.insert(0, str(work / 'src'))
import torch
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.experiment import score_windows
from openjev_dss.study_pair_visuals import GatedPairVisualStore
from openjev_dss.cloud_io import checked, sanitize

root = Path('/content/ucf_drive')
protocol = json.loads((work / 'research/protocol.json').read_text())
registry = json.loads((work / 'research/formal_inference_jobs.json').read_text())
group_name = os.environ['UCF_GROUP']
assert group_name == 'formal_mil'
jobs = registry['groups'][group_name]
deadline = float(os.environ['UCF_DEADLINE'])
cloud = 'gdrive:remaining_20261108_v1'
rc = ['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf', '--contimeout', '15s',
      '--timeout', '60s', '--retries', '3', '--low-level-retries', '10', '--retries-sleep', '2s']
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
started = time.time()
group_status = {'status': 'running', 'group': group_name, 'started': started, 'completed': []}
group_receipt = Path('/content/ucf_group_status.json')
group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')


def guard(reserve=180):
    if time.time() > deadline - reserve:
        raise TimeoutError('Stopping with time reserved for durable results and runtime release')


def transfer(path, remote, timeout=180):
    guard(90)
    checked(rc + ['copyto', str(path), remote], timeout=timeout)


def model_base(job):
    # Public models need no account consent; pin the exact historical/model-study revision.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(job['model_id'], revision=job['revision'], token=False,
                allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', '*.model'], max_workers=4))


records = json.loads((root / 'formal_stride1/manifest.json').read_text())['videos']
canonical = {r['name']: r for r in json.loads((root / 'formal_stride1/config.json').read_text())['canonical']}
assert len(records) == 290 and sum(r['n_frames'] for r in records) == 1111808
training = root / 'remaining_20261108_v1/formal_v2/training'
training_state = json.loads((training / 'status.json').read_text())
training_policy = json.loads((work / 'research/training_protocol_v2.json').read_text())
assert json.loads((training / 'policy.json').read_text()) == training_policy
assert training_state['status'] == 'complete' and training_state['test_used'] is False
for arm in training_state['arms'].values():
    assert arm['stopped'] and arm['stop_reason'] == 'validation_plateau'
    assert arm['frozen_parameters_unchanged']
    assert len(arm['validation']) >= training_policy['selection']['min_epochs']
    assert all(v['videos'] == training_policy['validation']['videos'] for v in arm['validation'])
runtime, active_mode, provider, gate_provider = None, None, None, None
for job in jobs:
    guard()
    out = Path('/content/ucf_results') / job['name']
    out.mkdir(parents=True, exist_ok=True)
    phase_started = time.time()
    descriptor = {**job, 'protocol_sha256': hashlib.sha256((work / 'research/protocol.json').read_bytes()).hexdigest(),
                  'torch': torch.__version__, 'batch': 4, 'dtype': 'bf16', 'prefix_cache': False,
                  'runtime_code_sha256': hashlib.sha256((work / 'src/openjev_dss/controlled_runtime.py').read_bytes()).hexdigest(),
                  'visual_code_sha256': hashlib.sha256((work / 'src/openjev_dss/study_visuals.py').read_bytes()).hexdigest(),
                  'pair_visual_code_sha256': hashlib.sha256((work / 'src/openjev_dss/study_pair_visuals.py').read_bytes()).hexdigest(),
                  'inference_code_sha256': hashlib.sha256((work / 'research/run_formal_inference.py').read_bytes()).hexdigest()}
    arm = training_state['arms'][job['training_branch']]
    selected = arm['selected_artifact']
    assert selected['epoch'] == arm['best_epoch']
    selection = json.loads((training / job['training_branch'] / 'selection.json').read_text())
    assert selection['artifact'] == selected and selection['test_used'] is False
    state_path = (training / selected['state_file']).resolve()
    adapter_path = (training / selected['adapter']).resolve()
    assert state_path.is_relative_to(training.resolve()) and adapter_path.is_relative_to(training.resolve())
    checkpoint_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert checkpoint_sha == selected['state_file_sha256']
    descriptor.update(checkpoint_sha256=checkpoint_sha, selected_epoch=arm['best_epoch'],
        training_fingerprint=training_state['fingerprint'],
        training_status_sha256=hashlib.sha256((training / 'status.json').read_bytes()).hexdigest(),
        training_policy_sha256=hashlib.sha256((work / 'research/training_protocol_v2.json').read_bytes()).hexdigest(),
        training_contract_sha256=hashlib.sha256((training / 'contract.json').read_bytes()).hexdigest())
    if job['mode'].startswith('openjev'):
        descriptor['adapter_revision'] = '0c7aa498b1627be8da4acf34c863ff0ee0a92785'
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    remote_dir = cloud + '/jobs/' + job['name']
    durable_complete = root / 'remaining_20261108_v1/jobs' / job['name'] / 'complete.json'
    if durable_complete.exists():
        old_complete = json.loads(durable_complete.read_text())
        assert old_complete == {'complete': True, 'fingerprint': fingerprint, 'videos': 290, 'windows': 37188, 'frames': 1111808}
        group_status['completed'].append(job['name'])
        group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')
        print('ALREADY_DURABLE', job['name'], flush=True)
        continue
    resumed = root / 'remaining_20261108_v1/jobs' / job['name'] / 'partial.zip'
    if resumed.exists() and not (out / 'config.json').exists():
        with zipfile.ZipFile(resumed) as archive:
            old_config = json.loads(archive.read('config.json'))
            if old_config['fingerprint'] != fingerprint:
                assert not any(name.startswith('predictions/') and name.endswith('.json') for name in archive.namelist()), 'Code changed after actual scores; use a new explicit run version'
                # Preserve a failed setup attempt; no test predictions existed to reuse.
                preserved = remote_dir + '/failed_empty_' + old_config['fingerprint'][:12] + '.zip'
                transfer(resumed, preserved)
                (out / 'superseded_empty_attempt.json').write_text(json.dumps({'fingerprint': old_config['fingerprint'], 'archive': preserved}) + '\n')
            else:
                for entry in archive.namelist():
                    assert (out / entry).resolve().is_relative_to(out.resolve())
                archive.extractall(out)
    if (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        assert old['fingerprint'] == fingerprint, 'Cannot mix outputs from different code/configuration'
    (out / 'config.json').write_text(json.dumps({'fingerprint': fingerprint, **descriptor}, indent=2) + '\n')
    status = {'status': 'running', 'job': job['name'], 'fingerprint': fingerprint, 'videos': 0,
              'gpu': torch.cuda.get_device_name(), 'started': phase_started, 'vision_or_cache_s': 0., 'language_s': 0.}
    predictions = out / 'predictions'
    predictions.mkdir(exist_ok=True)

    def snapshot(final=False):
        status['elapsed_s'] = time.time() - phase_started
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        group_status.update(active_job=job['name'], active_status=status, updated=time.time())
        group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')
        archive = out.parent / (job['name'] + '.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(out).as_posix())
        transfer(archive, remote_dir + ('/results.zip' if final else '/partial.zip'))
        if final:
            transfer(out / 'complete.json', remote_dir + '/complete.json')

    try:
        if active_mode != job['mode']:
            if runtime is not None:
                provider, gate_provider = None, None
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
            base = model_base(job)
            published = root / 'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint'
            options = {'state': state_path}
            if job['training_branch'] == 'openjev':
                options['published_checkpoint'] = published
            else:
                options['adapter'] = str(adapter_path)
            runtime = ControlledRuntime(str(base), protocol['prompts'][job['prompt']], job['duration_s'], **options)
            active_mode = job['mode']
        else:
            runtime.set_prompt(protocol['prompts'][job['prompt']], job['duration_s'])
        runtime.model.eval()
        runtime.head.eval()
        status['template_sha256'] = runtime.template_sha256
        torch.cuda.reset_peak_memory_stats()
        cache_remote = root / 'remaining_20261108_v1/visual_features'
        def sync_feature(path):
            transfer(path, cloud + '/visual_features/' + path.name, timeout=180)
        provider = GatedPairVisualStore(runtime, root / 'formal_stride1/canonical', canonical,
                                     '/content/ucf_visual_cache', job['revision'], cache_remote,
                                     sync=sync_feature, guard=guard)
        # Admit optimization using training videos, never by inspecting test labels/scores.
        train_manifest = json.loads((root / 'weak_dss_learning_10cu_v1/manifest.json').read_text())
        fixtures = [min((r for r in train_manifest['records'] if r['split'] == 'train' and r['label'] == y),
                        key=lambda r: (r['size'], r['name'])) for y in [0, 1]]
        fixture_metadata = {}
        for r in fixtures:
            m = json.loads((root / 'weak_dss_smoke_v1/metadata' / (r['name'] + '.json')).read_text())
            fixture_metadata[r['name']] = {**m, 'source_sha256': r['sha256']}
        gate_provider = GatedPairVisualStore(runtime, root / 'weak_dss_smoke_v1/canonical', fixture_metadata,
                '/content/ucf_gate_cache', job['revision'], guard=guard)
        provider.pair_gate = gate_provider.gate(fixtures, job['duration_s'])
        status['pair_feature_gate'] = provider.pair_gate
        for i, record in enumerate(records):
            guard()
            target = predictions / (record['name'] + '.json')
            if target.exists():
                p = json.loads(target.read_text())
                assert p['fingerprint'] == fingerprint and p['source_sha256'] == record['sha256']
                assert len(p['raw_logits']) == math.ceil(record['n_frames'] / record['fps'])
            else:
                start = time.time()
                features = provider.get(record, job['duration_s'])
                status['vision_or_cache_s'] += time.time() - start
                assert features.shape[1] == runtime.mask.sum().item()
                torch.cuda.synchronize()
                start = time.time()
                raw = score_windows(runtime.forward, features, 4).tolist()
                torch.cuda.synchronize()
                status['language_s'] += time.time() - start
                assert len(raw) == math.ceil(record['n_frames'] / record['fps']) and all(math.isfinite(v) for v in raw)
                p = {'name': record['name'], 'fingerprint': fingerprint, 'source_sha256': record['sha256'],
                     'canonical_sha256': canonical[record['name']]['canonical_sha256'],
                     'start_s': list(range(len(raw))), 'raw_logits': raw}
                temporary = target.with_suffix('.partial')
                temporary.write_text(json.dumps(p))
                temporary.replace(target)
                del features
            status.update(videos=i + 1, current_video=record['name'])
            print(job['name'], i + 1, '/290', record['name'], flush=True)
            if (i + 1) % 10 == 0:
                snapshot()
        assert len(list(predictions.glob('*.json'))) == 290
        windows = sum(len(json.loads(p.read_text())['raw_logits']) for p in predictions.glob('*.json'))
        assert windows == 37188
        status.update(status='complete', windows=windows, peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
        (out / 'complete.json').write_text(json.dumps({'complete': True, 'fingerprint': fingerprint,
                'videos': 290, 'windows': windows, 'frames': 1111808}, indent=2) + '\n')
        snapshot(final=True)
        group_status['completed'].append(job['name'])
        group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')
    except Exception:
        status.update(status='error', error=sanitize(traceback.format_exc()))
        try:
            snapshot()
        except Exception:
            pass
        group_status.update(status='error', error=status['error'], updated=time.time())
        group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')
        raise
group_status.update(status='complete', updated=time.time(), elapsed_s=time.time() - started)
group_receipt.write_text(json.dumps(group_status, indent=2) + '\n')
transfer(group_receipt, cloud + '/groups/' + group_name + '.json')
print('INFERENCE_GROUP_COMPLETE', group_name, time.time() - started, flush=True)
