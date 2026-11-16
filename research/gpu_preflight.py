"""Train-fixture-only BF16/gradient/throughput probe, bounded by a 15-minute watchdog."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

begin = time.time()
work = Path('/content/ucf_project')
work.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/ucf_project.zip') as archive:
    archive.extractall(work)
deps = Path('/content/ucf_deps')
deps.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--target', str(deps),
                'transformers==5.10.2', 'peft==0.19.1', 'tokenizers==0.22.2',
                'av==19.0.0', 'einops==0.8.2', 'fla-core==0.5.2'], check=True, timeout=240)
sys.path.insert(0, str(deps))
sys.path.insert(0, str(work / 'src'))
report = {'status': 'running', 'started': begin, 'test_labels_used': False}
destination = Path('/content/ucf_gpu_preflight.json')

def persist():
    destination.write_text(json.dumps(report, indent=2) + '\n')

try:
    import numpy as np
    import torch
    from openjev_dss.controlled_runtime import ControlledRuntime
    from openjev_dss.runtime import VisualStore, decode_canonical_frames
    from openjev_dss.losses import loss
    from openjev_dss.training import two_pass
    torch.set_num_threads(2)
    root = Path('/content/ucf_drive')
    protocol = json.loads((work / 'research/protocol.json').read_text())
    manifest = json.loads((root / 'weak_dss_learning_10cu_v1/manifest.json').read_text())
    report.update(gpu=torch.cuda.get_device_name(), memory_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3,
                  versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'fla-core', 'triton']},
                  stage='copy_base')
    persist()
    subprocess.run(['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf', 'copy',
                    'gdrive:models/base/15852e8c16360a2fea060d615a32b45270f8a8fc', '/content/ucf_base',
                    '--transfers', '4', '--checkers', '4', '--contimeout', '15s', '--timeout', '60s'], check=True, timeout=300)
    report['stage'] = 'load_model'
    persist()
    rt = ControlledRuntime('/content/ucf_base', protocol['prompts']['P1'], train=True)
    report.update(stage='train_fixture_gate', trainable_parameters=rt.trainable_parameters,
                  template_sha256=rt.template_sha256)
    persist()
    old = torch.load(root / 'formal_stride1/template.pt', map_location='cpu', weights_only=True)
    for field in ['input_ids', 'attention_mask', 'mm_token_type_ids', 'video_grid_thw']:
        torch.testing.assert_close(rt.template[field].cpu(), old[field][:1], rtol=0, atol=0)
    fixtures = [min((r for r in manifest['records'] if r['split'] == 'train' and r['label'] == label),
                    key=lambda r: (r['size'], r['name'])) for label in [0, 1]]
    report['fixture_names'] = [r['name'] for r in fixtures]
    windows = []
    provider = VisualStore(rt, root / 'weak_dss_smoke_v1', local_cache='/content/ucf_fixture_cache')
    report['historical_visual_cache_gate'] = []
    for r in fixtures:
        metadata, starts, signature, path = provider.descriptor(r, 0, False)
        pixels = decode_canonical_frames(root / 'weak_dss_smoke_v1/canonical' / r['name'],
                                         metadata['sampled_frames'], starts[:2], False)
        selected = [np.stack([pixels[min(2 * start + j, metadata['sampled_frames'])] for j in range(8)]) for start in starts[:2]]
        features = rt.vision(selected)
        assert torch.isfinite(features).all()
        if path.exists():
            saved = provider.cache.get(path, signature)
            difference = float((features.float() - saved[:len(features)].float()).abs().max())
            report['historical_visual_cache_gate'].append({'name': r['name'], 'exact': difference == 0., 'max_abs': difference})
        windows.append(features)
    report['cache_reuse_allowed'] = bool(report['historical_visual_cache_gate']) and all(r['exact'] for r in report['historical_visual_cache_gate'])
    rt.model.train()
    params = [p for p in rt.model.parameters() if p.requires_grad]
    packed = torch.cat(windows)
    physical = torch.cat([packed, packed[:1].expand(4 - len(packed), *packed.shape[1:])]) if len(packed) < 4 else packed
    dense_scores = rt.forward(physical).float()[:len(packed)]
    dense_objective, _ = loss(list(dense_scores.split([len(w) for w in windows])), [0, 1], False)
    dense_objective.backward()
    dense_gradient = torch.cat([p.grad.float().flatten() for p in params])
    rt.model.zero_grad(set_to_none=True)
    two_pass(rt.forward, windows, [0, 1], False, chunk=4, score_batch=4, pad_batches=True)
    leaf_gradient = torch.cat([p.grad.float().flatten() for p in params])
    relative = float(torch.linalg.vector_norm(dense_gradient - leaf_gradient) / torch.linalg.vector_norm(dense_gradient))
    cosine = float(torch.nn.functional.cosine_similarity(dense_gradient.double(), leaf_gradient.double(), dim=0))
    assert relative < .01 and cosine > .999 and torch.isfinite(leaf_gradient).all()
    report['gradient_gate'] = {'relative_l2': relative, 'cosine': cosine, 'passed': True}
    rt.model.zero_grad(set_to_none=True)
    rt.model.eval()
    calls = {}
    for module in rt.model.modules():
        if hasattr(module, 'chunk_gated_delta_rule'):
            fn = module.chunk_gated_delta_rule
            label = fn.__module__ + '.' + fn.__name__
            def count(*args, _fn=fn, _label=label, **kwargs):
                calls[_label] = calls.get(_label, 0) + 1
                return _fn(*args, **kwargs)
            module.chunk_gated_delta_rule = count
    with torch.inference_mode():
        for _ in range(2):
            rt.forward(physical)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            score = rt.forward(physical)
        torch.cuda.synchronize()
    report['language_windows_per_second'] = 10 * len(physical) / (time.time() - start)
    report['gated_delta_calls'] = calls
    assert any(name.startswith('fla.') for name in calls), 'Expected FLA kernels were not executed'
    report.update(status='passed', stage='complete', peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  elapsed_s=time.time() - begin)
except Exception:
    report.update(status='error', error=traceback.format_exc(), elapsed_s=time.time() - begin)
finally:
    persist()
    subprocess.run(['/content/ucf-bin/rclone', '--config', '/content/ucf-rclone.conf', 'copyto',
                    str(destination), 'gdrive:remaining_20261108_v1/preflight/gpu_preflight.json'], timeout=90)
print(json.dumps(report, indent=2))
