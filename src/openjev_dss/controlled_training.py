"""Single-arm Qwen MIL run with the historical split, schedule and optimizer budget."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .experiment import score_windows
from .learning import atomic_json, schedule
from .training import restore, trainable_state, two_pass


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_single_mil(runtime, manifest, provider, out, deadline, sync=lambda: None):
    """One complete 160-step epoch, then full validation; never reads test labels."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    plan = schedule(manifest['records'], 1)
    assert len(plan) == 160 and manifest['audit']['near_duplicate_visual_audit'] == 'passed'
    params = [p for p in runtime.model.parameters() if p.requires_grad]
    assert params and not list(runtime.head.parameters()), 'Native Qwen readout must stay frozen'
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
    frozen_versions = [(p, p._version) for p in runtime.full.parameters() if not p.requires_grad]
    initial = tensor_digest(trainable_state(runtime.model, runtime.head))
    status = {'status': 'training', 'steps': 0, 'epoch_target': 1, 'steps_target': 160,
              'manifest_fingerprint': config, 'rows': [], 'started': time.time(),
              'initial_lora_sha256': initial, 'generation': -1, 'test_used': False}
    if (out / 'status.json').exists():
        status = json.loads((out / 'status.json').read_text())
        assert status['manifest_fingerprint'] == config
        if status['status'] == 'complete':
            return status
        resume = torch.load(out / f"resume_{status['generation'] % 2}.pt", map_location='cpu', weights_only=True)
        assert resume['steps'] == status['steps']
        restore(runtime.model, runtime.head, resume['trainable'])
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['rng'])
        torch.cuda.set_rng_state(resume['cuda_rng'])

    def persist():
        previous_generation = status['generation']
        generation = previous_generation + 1
        target = out / f'resume_{generation % 2}.pt'
        tmp = target.with_suffix('.tmp')
        torch.save({'steps': status['steps'], 'trainable': trainable_state(runtime.model, runtime.head),
                    'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state()}, tmp)
        tmp.replace(target)
        status['generation'] = generation
        status['elapsed_s'] = time.time() - status['started']
        atomic_json(out / 'status.json', status)
        try:
            sync()
        except Exception:
            # Retry the SAME slot after a failed publish. Advancing again could
            # overwrite the other slot still referenced by the durable status.
            status['generation'] = previous_generation
            raise

    def guard():
        if time.time() > deadline - 180:
            raise TimeoutError('Budget reserve: state will persist before runtime release')

    try:
        for step in range(status['steps'], len(plan)):
            guard()
            epoch, pairs = plan[step]
            windows = [provider(r, epoch, False) for pair in pairs for r in pair]
            guard()
            runtime.model.train()
            optimizer.zero_grad(set_to_none=True)
            begin = time.time()
            objective, info = two_pass(runtime.forward, windows, [0, 1] * len(pairs), False,
                                       chunk=4, score_batch=4, pad_batches=True)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            status['steps'] = step + 1
            status['rows'].append({'step': step + 1, 'loss': objective, 'stats': info,
                                   'grad_norm': float(norm), 'seconds': time.time() - begin,
                                   'bag_windows': [len(w) for w in windows],
                                   'training_instances': sum(len(w) for w in windows),
                                   'videos': [r['name'] for pair in pairs for r in pair]})
            if (step + 1) % 10 == 0:
                persist()
        status['status'] = 'validation'
        persist()
        runtime.model.eval()
        runtime.head.eval()
        validation = [r for r in manifest['records'] if r['split'] == 'val']
        path = out / 'validation.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        assert [r['name'] for r in rows] == [r['name'] for r in validation[:len(rows)]]
        for record in validation[len(rows):]:
            guard()
            features = provider(record, 0, True)
            parts = []
            for i in range(0, len(features), 4):
                guard()
                parts.append(score_windows(runtime.forward, features[i:i + 4], 4))
            probabilities = torch.cat(parts).sigmoid()
            value = float(probabilities.topk(max(1, math.ceil(len(probabilities) / 8))).values.mean())
            rows.append({'name': record['name'], 'label': record['label'], 'probability': value,
                         'windows': len(probabilities)})
            atomic_json(path, rows)
            if len(rows) % 10 == 0:
                sync()
        assert len(rows) == 163
        ap = float(average_precision_score([r['label'] for r in rows], [r['probability'] for r in rows]))
        current = trainable_state(runtime.model, runtime.head)
        assert tensor_digest(current) != status['initial_lora_sha256'], 'LoRA did not update'
        assert all(p._version == version for p, version in frozen_versions), 'Frozen parameter changed'
        torch.save(current, out / 'best_state.pt')
        runtime.model.save_pretrained(out / 'best_adapter')
        atomic_json(out / 'selection.json', {'epoch': 1, 'step': 160, 'video_ap': ap, 'test_used': False})
        status.update(status='complete', validation_video_ap=ap, validation_videos=163,
                      final_lora_sha256=tensor_digest(current), frozen_versions_unchanged=True,
                      best_state_sha256=hashlib.sha256((out / 'best_state.pt').read_bytes()).hexdigest())
    except TimeoutError:
        status['status'] = 'budget_stopped'
    except Exception:
        import traceback
        status.update(status='error', error=traceback.format_exc())
    finally:
        persist()
    return status
