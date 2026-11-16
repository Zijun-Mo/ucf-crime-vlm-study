"""Paired multi-epoch MIL with validation-only stopping and durable checkpoints."""
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from .cloud_io import sanitize
from .convergence import record_validation, convergence_summary
from .controlled_training import tensor_digest
from .experiment import score_windows
from .learning import atomic_json, schedule
from .training import restore, trainable_state
from .formal_backward import two_pass


def _cuda_rng(runtime):
    return torch.cuda.get_rng_state() if str(runtime.device).startswith('cuda') else None


def _set_rng(runtime, rng, cuda_rng):
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)


def run_formal_mil(runtimes, manifest, provider, out, deadline, policy, epoch_limit, sync=lambda: None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assert set(runtimes) == {'qwen', 'openjev'}
    assert manifest['audit']['near_duplicate_visual_audit'] == 'passed'
    fingerprint = hashlib.sha256(json.dumps({'manifest': manifest, 'policy': policy}, sort_keys=True).encode()).hexdigest()
    steps_per_epoch = policy['training']['steps_per_epoch']
    plan = schedule(manifest['records'], epoch_limit)
    assert len(plan) == steps_per_epoch * epoch_limit
    validation = [r for r in manifest['records'] if r['split'] == 'val']
    assert len(validation) == policy['validation']['videos']
    state = {'status': 'running', 'stage': 'training', 'epoch_cursor': 0, 'step_cursor': 0,
             'generation': -1, 'fingerprint': fingerprint, 'started': time.time(),
             'epoch_limit': epoch_limit, 'test_used': False, 'arms': {}, 'feature_rows': []}
    optimizers, parameters, rngs, frozen = {}, {}, {}, {}
    for name, rt in runtimes.items():
        lora = [p for p in rt.model.parameters() if p.requires_grad]
        head = [p for p in rt.head.parameters() if p.requires_grad]
        assert lora and (name != 'qwen' or not head)
        groups = [{'params': lora, 'lr': policy['optimizer']['lora_lr']}]
        if head:groups.append({'params': head, 'lr': policy['optimizer']['scalar_head_lr']})
        optimizers[name] = torch.optim.AdamW(groups, weight_decay=policy['optimizer']['weight_decay'])
        parameters[name] = lora + head
        frozen[name] = [(p, p._version) for p in rt.full.parameters() if not p.requires_grad]
        rngs[name] = {'rng': torch.get_rng_state(), 'cuda_rng': _cuda_rng(rt)}
        state['arms'][name] = {'steps': 0, 'rows': [], 'validation': [], 'stopped': False,
            'initial_digest': tensor_digest(trainable_state(rt.model, rt.head))}
    status_path = out/'status.json'
    durable_steps = None
    if status_path.exists():
        state = json.loads(status_path.read_text())
        assert state['fingerprint'] == fingerprint, 'Formal scientific training contract changed'
        assert epoch_limit >= state['epoch_cursor']
        if state['status'] == 'complete':return state
        state.update(status='running', epoch_limit=epoch_limit)
        if state.get('error'):
            state.setdefault('past_errors', []).append(state.pop('error'))
        for name, rt in runtimes.items():
            saved = torch.load(out/name/f"resume_{state['generation']%2}.pt", map_location='cpu', weights_only=True)
            assert saved['steps'] == state['arms'][name]['steps'] == len(state['arms'][name]['rows'])
            restore(rt.model, rt.head, saved['trainable'])
            optimizers[name].load_state_dict(saved['optimizer'])
            rngs[name] = {'rng': saved['rng'], 'cuda_rng': saved['cuda_rng']}
        durable_steps = {n: a['steps'] for n, a in state['arms'].items()}

    def guard(reserve=180):
        if time.time() > deadline-reserve:
            raise TimeoutError('Formal training allocation budget reserve')

    def publish_status():
        state['elapsed_s'] = time.time()-state['started']
        atomic_json(status_path, state)
        sync()

    def persist():
        nonlocal durable_steps
        steps = {n: a['steps'] for n, a in state['arms'].items()}
        if durable_steps == steps and state['generation'] >= 0:
            publish_status(); return
        previous = state['generation']; generation = previous+1
        for name, rt in runtimes.items():
            folder = out/name; folder.mkdir(exist_ok=True)
            path = folder/f'resume_{generation%2}.pt'; tmp = path.with_suffix('.tmp')
            torch.save({'steps': steps[name], 'trainable': trainable_state(rt.model, rt.head),
                        'optimizer': optimizers[name].state_dict(), **rngs[name]}, tmp)
            tmp.replace(path)
        state['generation'] = generation
        try:publish_status()
        except Exception:
            state['generation'] = previous
            raise
        durable_steps = steps

    def validate(name, epoch):
        rt = runtimes[name]; rt.model.eval(); rt.head.eval()
        digest = tensor_digest(trainable_state(rt.model, rt.head))
        path = out/name/f'epoch_{epoch}_validation.json'
        data = {'fingerprint': fingerprint, 'state_sha256': digest, 'rows': []}
        if path.exists():
            data = json.loads(path.read_text())
            assert data['fingerprint'] == fingerprint and data['state_sha256'] == digest
        rows = data['rows']
        assert [r['name'] for r in rows] == [r['name'] for r in validation[:len(rows)]]
        for record in validation[len(rows):]:
            guard(); features = provider(record, epoch-1, True); parts = []
            for i in range(0, len(features), 4):
                guard(); parts.append(score_windows(rt.forward, features[i:i+4], 4))
            z = torch.cat(parts)
            assert torch.isfinite(z).all()
            rows.append({'name': record['name'], 'label': record['label'], 'max_logit': float(z.max()),
                         'topk_probability': float(z.sigmoid().topk(max(1, math.ceil(len(z)/8))).values.mean()),
                         'windows': len(z)})
            atomic_json(path, data)
            del features,parts,z
            if len(rows)%10 == 0:
                state.setdefault('validation_progress', {})[name] = len(rows)
                publish_status()
        assert len(rows) == len(validation)
        y = [r['label'] for r in rows]; z = [r['max_logit'] for r in rows]
        return {'video_max_ap': float(average_precision_score(y,z)), 'video_max_auroc': float(roc_auc_score(y,z)),
                'state_sha256': digest, 'videos': len(rows)}

    def save_epoch(name, epoch):
        rt = runtimes[name]; folder = out/name/'epochs'/f'epoch_{epoch}'
        folder.mkdir(parents=True, exist_ok=True)
        state_path = folder/'state.pt'; tmp = state_path.with_suffix('.tmp')
        torch.save(trainable_state(rt.model, rt.head), tmp); tmp.replace(state_path)
        rt.model.save_pretrained(folder/'adapter')
        torch.save(rt.head.state_dict(), folder/'head.pt')
        return {'epoch': epoch, 'state_file': str(state_path.relative_to(out)),
                'state_file_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
                'adapter': str((folder/'adapter').relative_to(out)), 'head': str((folder/'head.pt').relative_to(out))}

    try:
        while state['epoch_cursor'] < epoch_limit and not all(a['stopped'] for a in state['arms'].values()):
            ep = state['epoch_cursor']
            if state['stage'] == 'training':
                for step in range(state['step_cursor'], steps_per_epoch):
                    guard(); actual_epoch, pairs = plan[ep*steps_per_epoch+step]
                    assert actual_epoch == ep
                    start = time.time(); bags = [provider(r,ep,False) for pair in pairs for r in pair]
                    feature_s = time.time()-start
                    absolute_step = ep*steps_per_epoch+step
                    for name, rt in runtimes.items():
                        arm = state['arms'][name]
                        if arm['stopped'] or arm['steps'] > absolute_step:continue
                        assert arm['steps'] == absolute_step
                        guard(); _set_rng(rt, **rngs[name]); rt.model.train(); rt.head.train()
                        opt = optimizers[name]; opt.zero_grad(set_to_none=True); start = time.time()
                        objective, info = two_pass(rt.forward,bags,[0,1]*len(pairs),False,chunk=4,score_batch=4,pad_batches=True)
                        norm = torch.nn.utils.clip_grad_norm_(parameters[name],policy['optimizer']['clip_grad_norm'])
                        assert torch.isfinite(norm); opt.step()
                        rngs[name] = {'rng':torch.get_rng_state(),'cuda_rng':_cuda_rng(rt)}
                        arm['steps'] += 1
                        arm['rows'].append({'step':arm['steps'],'epoch':ep+1,'loss':objective,'stats':info,
                            'grad_norm':float(norm),'seconds':time.time()-start,'bag_windows':[len(w) for w in bags],
                            'videos':[r['name'] for pair in pairs for r in pair]})
                    state['feature_rows'].append({'epoch':ep+1,'step':step+1,'seconds':feature_s})
                    state['step_cursor'] = step+1
                    if (step+1)%10 == 0:persist()
                state['stage'] = 'validation'; persist()
            if state['stage'] == 'validation':
                for name in runtimes:
                    arm = state['arms'][name]
                    if arm['stopped'] or len(arm['validation']) > ep:continue
                    metrics = validate(name,ep+1)
                    selected, stopped = record_validation(arm,ep+1,metrics['video_max_ap'],policy['selection'])
                    arm['validation'][-1].update(metrics)
                    if selected or ep == 0:
                        artifact = save_epoch(name,ep+1)
                        arm.setdefault('epoch_artifacts',{})[str(ep+1)] = artifact
                    if selected:
                        arm['selected_artifact'] = arm['epoch_artifacts'][str(ep+1)]
                    atomic_json(out/name/'selection.json',{'best_epoch':arm['best_epoch'],'video_max_ap':arm['best_ap'],
                        'artifact':arm['selected_artifact'],'test_used':False,'stop_reason':arm.get('stop_reason')})
                    persist()
                state.update(epoch_cursor=ep+1,step_cursor=0,stage='training'); persist()
        state['status'] = convergence_summary(state['arms'],epoch_limit)
        if state['status'] == 'complete':
            for name, rt in runtimes.items():
                assert all(p._version == version for p,version in frozen[name]), 'Frozen parameter changed'
                arm = state['arms'][name]
                assert tensor_digest(trainable_state(rt.model,rt.head)) != arm['initial_digest']
                arm['frozen_parameters_unchanged'] = True
    except TimeoutError:
        state['status'] = 'budget_stopped'
    except Exception:
        state.update(status='error',error=sanitize(traceback.format_exc()))
    finally:
        persist()
    return state
