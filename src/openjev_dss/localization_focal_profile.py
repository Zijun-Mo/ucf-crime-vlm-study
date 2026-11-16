"""Actual L4 derivative check and <=3h epoch forecast for the V3 objective."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from .learning import atomic_json
from .localization_backward import padded_forward,stream_scores
from .localization_focal import backward,terms
from .localization_pointwise import window_budget


def profile_focal(runtime,provider,records,policy,out,staging_seconds,rate):
    out=Path(out);source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('localization_*.py')}
    if (out/'throughput_profile.json').exists():
        old=json.loads((out/'throughput_profile.json').read_text())
        assert old['source']==source and old['qualified'] and old['gpu']==torch.cuda.get_device_name()
        return old
    assert torch.cuda.get_device_name()=='NVIDIA L4'
    batch=policy['input']['batch'];params=[p for p in runtime.model.parameters() if p.requires_grad]
    def clear():runtime.model.zero_grad(set_to_none=True)
    def vector():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    def sampled(n,label):
        return dict(indices=list(range(n)),positive=[1/n if label else 0.]*n,
            negative=[.5/n if label else 1/n]*n,total_windows=n,positive_draws=n if label else 0,
            background_draws=n if label else 0)
    train=[r for r in records if r['split']=='train'];checks=[];measurements=[]
    fixture=min(train,key=lambda r:(r['size'],r['name']));features=provider(fixture,0,True)[:8]
    for label in [0,1]:
        runtime.model.train();runtime.head.train();clear();plan=sampled(len(features),label)
        z=torch.cat([padded_forward(runtime.forward,features[i:i+batch],batch) for i in range(0,len(features),batch)])
        loss=terms(z,plan['positive'],plan['negative'],8.,2.).sum();loss.backward();expected=vector().clone();clear()
        backward(runtime.forward,features,plan,batch=batch,positive_weight=8.);actual=vector()
        relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-12));cosine=float(torch.nn.functional.cosine_similarity(actual.double(),expected.double(),dim=0))
        checks.append(dict(label=label,relative_l2=relative,cosine=cosine,passed=relative<.01 and cosine>.999));clear()
    gate=dict(passed=all(r['passed'] for r in checks),checks=checks,kind='pointwise_focal_v3',train_only=True)
    atomic_json(out/'gradient_preflight.json',gate)
    if not gate['passed']:raise RuntimeError('V3 real LoRA derivative gate failed; no optimizer updates')
    for label in [0,1]:
        eligible=sorted((r for r in train if r['label']==label and 64<=provider.metadata[r['name']]['window_count']<=1024),key=lambda r:(provider.metadata[r['name']]['window_count'],r['name']))
        for quantile in [.25,.75]:
            record=eligible[int(quantile*(len(eligible)-1))];features=provider(record,0,True)[:128]
            runtime.model.eval();runtime.head.eval()
            with runtime.model.disable_adapter():stream_scores(runtime.forward,features[:batch],batch)
            torch.cuda.synchronize();begin=time.time()
            with runtime.model.disable_adapter():stream_scores(runtime.forward,features,batch)
            torch.cuda.synchronize();forward_s=time.time()-begin
            runtime.model.train();runtime.head.train();clear()
            backward(runtime.forward,features[:batch],sampled(batch,label),batch=batch,positive_weight=8.);clear()
            torch.cuda.synchronize();begin=time.time()
            _,info=backward(runtime.forward,features,sampled(len(features),label),batch=batch,positive_weight=8.)
            torch.cuda.synchronize();seconds=time.time()-begin;clear()
            measurements.append(dict(name=record['name'],label=label,windows=len(features),physical_windows=info['physical_windows'],seconds=seconds,inference_seconds=forward_s))
    config={**policy['sampling'],'anchor_cap':policy['sampling']['background_cap']}
    budget=window_budget(records,provider.metadata,config,batch)
    train_s=sum(budget['by_class'][y]*max(r['seconds']/r['physical_windows'] for r in measurements if r['label']==y) for y in [0,1])
    val_s=budget['validation_physical_windows']*max(r['inference_seconds']/r['physical_windows'] for r in measurements)
    bound=(train_s+val_s)*1.3/3600+policy['resources']['epoch_io_reserve_hours']
    report=dict(qualified=bound<=3.,gpu=torch.cuda.get_device_name(),source=source,measurements=measurements,
        window_budget=budget,gradient_gate=True,epoch_upper_bound_hours=bound,epoch_max_hours=3.,
        epoch_training_only_hours=train_s/3600,validation_hours=val_s/3600,safety_factor=1.3,
        io_reserve_hours=policy['resources']['epoch_io_reserve_hours'],rate_cu_hr=rate,canonical_staging_s=staging_seconds,
        kind='pointwise_focal_v3',test_used=False,full_training_started=False)
    atomic_json(out/'throughput_profile.json',report);atomic_json(out/'optimization_qualification.json',report)
    if not report['qualified']:raise RuntimeError('V3 complete epoch bound exceeds user three-hour limit; no optimizer updates')
    return report
