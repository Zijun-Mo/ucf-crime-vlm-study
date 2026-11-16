"""Measured derivative and conservative epoch gate before any optimizer update."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .learning import atomic_json
from .localization_backward import padded_forward, stream_scores
from .localization_objective import selection
from .localization_pointwise import plan, terms, backward_pointwise, window_budget


def profile_pointwise(runtime,provider,records,policy,out,staging_seconds,rate):
    out=Path(out);profile_path=out/'throughput_profile.json'
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('localization_*.py')}
    if profile_path.exists():
        profile=json.loads(profile_path.read_text())
        assert profile['source']==source and profile['gpu']==torch.cuda.get_device_name()
        assert profile['qualified'] and profile['epoch_upper_bound_hours']<=policy['resources']['epoch_max_hours']
        return profile
    batch=policy['input']['batch'];params=[p for p in runtime.model.parameters() if p.requires_grad]
    def clear():runtime.model.zero_grad(set_to_none=True)
    def vector():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    train=[r for r in records if r['split']=='train']
    fixture=min((r for r in train if r['label']),key=lambda r:(r['size'],r['name']))
    view=provider(fixture,0,True);features=view[:min(8,len(view))]
    chosen=selection([-3.]*(len(features)//2)+[4.]*(len(features)-len(features)//2),dict(threshold=0.,scale=1.))
    runtime.model.train();runtime.head.train();checks=[]
    for label in [0,1]:
        sampled=plan(fixture['name'],label,len(features),chosen,0,
            dict(fraction=1.,normal_cap=64,positive_cap=32,anchor_cap=32))
        clear();indices=sampled['indices']
        z=torch.cat([padded_forward(runtime.forward,features[indices[i:i+batch]],batch) for i in range(0,len(indices),batch)])
        dense=terms(z,label,sampled['bce'],sampled['anchor'],sampled['teacher'],sampled['scale'],
            policy['loss']['anchor_weight'],policy['loss']['huber_delta']).sum()
        dense.backward();expected=vector().clone();value_dense=float(dense.detach());del z,dense
        clear();value,_=backward_pointwise(runtime.forward,features,sampled,batch=batch,
            anchor_weight=policy['loss']['anchor_weight'],huber_delta=policy['loss']['huber_delta'])
        actual=vector();relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(actual.double(),expected.double(),dim=0))
        checks.append(dict(label=label,relative_l2=relative,cosine=cosine,reference_loss=value_dense,
            stream_loss=value,passed=relative<.01 and cosine>.999))
        clear();del actual,expected
    gate=dict(passed=all(x['passed'] for x in checks),checks=checks,kind='pointwise_teacher_v2',train_only=True)
    atomic_json(out/'gradient_preflight.json',gate)
    if not gate['passed']:raise RuntimeError('Pointwise GPU derivative gate failed; no optimizer updates')
    measurements=[]
    for label in [0,1]:
        eligible=sorted((r for r in train if r['label']==label and 64<=provider.metadata[r['name']]['window_count']<=1024),
            key=lambda r:(provider.metadata[r['name']]['window_count'],r['name']))
        fixtures=[eligible[int(q*(len(eligible)-1))] for q in [.25,.75]]
        for record in fixtures:
            view=provider(record,0,True);features=view[:min(128,len(view))]
            runtime.model.eval();runtime.head.eval()
            with runtime.model.disable_adapter():stream_scores(runtime.forward,features[:batch],batch)
            torch.cuda.synchronize();started=time.time()
            with runtime.model.disable_adapter():teacher=stream_scores(runtime.forward,features,batch).tolist()
            torch.cuda.synchronize();forward_s=time.time()-started
            selected=selection(teacher,dict(threshold=float(np.median(teacher)),scale=1.))
            # Measure a worst-case dense combination of BCE and anchor on every
            # fixture row. Actual sparse plans never exceed this work per row.
            n=len(features);sampled=dict(indices=list(range(n)),bce=[1/n]*n,
                anchor=[1/n if label else 0.]*n,teacher=teacher,scale=1.,label=label,
                total_windows=n,positive_draws=n if label else 0,anchor_draws=n if label else 0)
            runtime.model.train();runtime.head.train();clear()
            warm={**sampled,'indices':list(range(batch))}
            backward_pointwise(runtime.forward,features,warm,batch=batch);clear()
            torch.cuda.synchronize();started=time.time()
            value,info=backward_pointwise(runtime.forward,features,sampled,batch=batch,
                anchor_weight=policy['loss']['anchor_weight'],huber_delta=policy['loss']['huber_delta'])
            torch.cuda.synchronize();seconds=time.time()-started;clear()
            measurements.append(dict(name=record['name'],label=label,windows=n,seconds=seconds,
                physical_windows=info['physical_windows'],inference_seconds=forward_s))
    budget=window_budget(records,provider.metadata,policy['sampling'],batch)
    per_class={y:max(x['seconds']/x['physical_windows'] for x in measurements if x['label']==y) for y in [0,1]}
    inference=max(x['inference_seconds']/(batch*((x['windows']+batch-1)//batch)) for x in measurements)
    training_s=sum(budget['by_class'][y]*per_class[y] for y in [0,1])
    validation_s=budget['validation_physical_windows']*inference
    bound=(training_s+validation_s)*policy['resources']['profile_safety_factor']/3600+policy['resources']['epoch_io_reserve_hours']
    total_teacher=sum(provider.metadata[r['name']]['window_count'] for r in records)*inference/3600
    profile=dict(status='passed' if bound<=policy['resources']['epoch_max_hours'] else 'speed_gate_failed',
        qualified=bound<=policy['resources']['epoch_max_hours'],kind='pointwise_teacher_v2',gpu=torch.cuda.get_device_name(),
        source=source,measurements=measurements,window_budget=budget,gradient_gate=True,
        epoch_training_only_hours=training_s/3600,validation_hours=validation_s/3600,
        epoch_upper_bound_hours=bound,epoch_max_hours=policy['resources']['epoch_max_hours'],
        safety_factor=policy['resources']['profile_safety_factor'],io_reserve_hours=policy['resources']['epoch_io_reserve_hours'],
        teacher_forward_only_hours=total_teacher,rate_cu_hr=rate,canonical_staging_s=staging_seconds,
        limitation='Conservative forecast from four train-only fixtures; includes physical tail padding, 30% compute contingency and 21min I/O allowance; actual epoch timing is monitored',
        test_used=False,full_training_started=False)
    atomic_json(profile_path,profile);atomic_json(out/'optimization_qualification.json',profile)
    if not profile['qualified']:raise RuntimeError('Pointwise complete epoch forecast exceeds three hours; training remains held')
    return profile
