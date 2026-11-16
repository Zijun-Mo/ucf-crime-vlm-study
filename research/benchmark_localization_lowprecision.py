"""Train-only throughput and gradient gates; never launches full training."""
import json,os,sys,time,traceback
from pathlib import Path
work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import numpy as np
import torch
from huggingface_hub import snapshot_download
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.formal_execution import apply_execution
from openjev_dss.formal_features import FormalVisualProvider
from openjev_dss.localization_objective import selection,objective
from openjev_dss.localization_backward import padded_forward,stream_scores
from openjev_dss.localization_fast_backward import backward_fixed_selection
from openjev_dss.learning import atomic_json

out=Path('/content/ucf_localization_optimization_precision');out.mkdir(exist_ok=True)
report=dict(status='running',stage='load',started=time.time(),candidates=[],full_training_started=False,test_used=False)
def persist():atomic_json(out/'status.json',report)
persist();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
try:
    base=snapshot_download('Qwen/Qwen3.5-2B',revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        token=False,allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'])
    protocol=json.loads((work/'research/protocol.json').read_text())
    runtime=ControlledRuntime(base,protocol['prompts']['P1'],train=True)
    manifest=json.loads(Path('/content/ucf_drive/weak_dss_learning_10cu_v1/manifest.json').read_text())
    records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    metadata=json.loads((work/'research/formal_training_metadata.json').read_text())['records']
    params=[p for p in runtime.model.parameters() if p.requires_grad]
    def clear():runtime.model.zero_grad(set_to_none=True)
    def vector():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    fixtures=[]
    inputs=torch.load('/content/localization_probe_fixtures.pt',map_location='cpu',weights_only=True)
    lookup={r['name']:r for r in records if r['split']=='train'}
    for item in sorted(inputs,key=lambda r:r['label']):
        record=lookup[item['name']];assert record['sha256']==item['source_sha256']
        features=item['features']
        runtime.model.eval();runtime.head.eval()
        with runtime.model.disable_adapter():teacher=stream_scores(runtime.forward,features).tolist()
        chosen=selection(teacher,dict(threshold=float(np.median(teacher)),scale=1.))
        fixtures.append((record,features,chosen))
    report['fixed_feature_input_note']='Two training-only fixed L4 token fixtures for hardware screening, not qualification of the full training data path'
    by_class={y:sum(metadata[r['name']]['window_count'] for r in records if r['split']=='train' and r['label']==y) for y in [0,1]}
    # Diagnose the unchanged path before changing precision. The fixtures are
    # resident CPU tensors, so this isolates launch/sync/compute from disk I/O.
    import subprocess,resource
    apply_execution({'qwen':runtime},dict(attention_backend='math',gradient_checkpointing=False))
    runtime.model.train();runtime.head.train()
    normal=fixtures[0][1]
    clear();backward_fixed_selection(runtime.forward,normal[:4],0,batch=4);clear()
    samples=open('/content/localization_gpu_samples.csv','w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits','--loop-ms=100'],stdout=samples,stderr=subprocess.DEVNULL)
    torch.cuda.synchronize();begin=time.perf_counter();cpu_begin=time.process_time()
    backward_fixed_selection(runtime.forward,normal,0,batch=4)
    torch.cuda.synchronize();wall=time.perf_counter()-begin;cpu=time.process_time()-cpu_begin;clear()
    monitor.terminate();monitor.wait(timeout=5);samples.close()
    readings=[]
    for line in Path('/content/localization_gpu_samples.csv').read_text().splitlines():
        try:readings.append([float(x.strip()) for x in line.split(',')[1:]])
        except ValueError:pass
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False) as prof:
        backward_fixed_selection(runtime.forward,normal[:16],0,batch=4)
        torch.cuda.synchronize()
    clear();trace='/content/localization_native_trace.json';prof.export_chrome_trace(trace)
    events=json.loads(Path(trace).read_text())['traceEvents']
    def interval_union(rows):
        merged=[]
        for a,b in sorted(rows):
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        return sum(b-a for a,b in merged)
    kernels=[(x['ts'],x['ts']+x['dur']) for x in events if x.get('cat')=='kernel' and 'dur' in x]
    gpu_span=max(b for _,b in kernels)-min(a for a,_ in kernels)
    ops=prof.key_averages()
    gpu_top=sorted(ops,key=lambda x:x.self_device_time_total,reverse=True)[:20]
    cpu_top=sorted(ops,key=lambda x:x.self_cpu_time_total,reverse=True)[:20]
    report['native_profile']={'resident_tensor_windows':len(normal),'wall_seconds':wall,'process_cpu_seconds':cpu,
        'gpu_util_samples':len(readings),'mean_gpu_util':float(np.mean([r[0] for r in readings])) if readings else None,
        'mean_memory_util':float(np.mean([r[1] for r in readings])) if readings else None,
        'mean_power_w':float(np.mean([r[2] for r in readings])) if readings else None,
        'profile_kernel_union_fraction':interval_union(kernels)/gpu_span,
        'gpu_top':[{'op':x.key,'self_cuda_us':x.self_device_time_total,'calls':x.count} for x in gpu_top],
        'cpu_top':[{'op':x.key,'self_cpu_us':x.self_cpu_time_total,'calls':x.count} for x in cpu_top],
        'note':'Profiler introduces CPU overhead; compare trace kernel occupancy with unprofiled utilization and timings. Not proof of a single bottleneck.'}
    persist()
    from openjev_dss.localization_precision import patch_frozen_linears,set_precision
    modules=patch_frozen_linears(runtime)
    report['patched_frozen_linears']=len(modules)
    report['precision_policy']={'train_only':True,'teacher_and_validation':'unchanged BF16',
        'authorized_by_user':True,'drift_gate':{'relative_l2_max':.1,'cosine_min':.99},
        'gradient_algebra_gate':{'relative_l2_max':.01,'cosine_min':.999}}
    baseline={}
    apply_execution({'qwen':runtime},dict(attention_backend='math',gradient_checkpointing=False))
    runtime.model.train();runtime.head.train()
    for label in [0,1]:
        clear();features=fixtures[label][1][:4]
        chosen=selection([-3.,-3.,4.,4.],dict(threshold=0.,scale=1.))
        z=padded_forward(runtime.forward,features,4);loss,_=objective(z,label,chosen);loss.backward()
        baseline[label]={'gradient':vector().cpu(),'logits':z.detach().cpu(),'loss':float(loss.detach())}
        clear();del z,loss
    for precision,backend,batch in [('bf16','math',4),('bf16_tf32','math',4),('bf16_reduced_math','math',4),('bf16_reduced_math','math',8),('fp8_forward','math',4),('fp8_full','math',4),('fp8_full','math',8),('fp8_full','math',16)]:
        set_precision(modules,precision if precision.startswith('fp8') else 'bf16')
        torch.set_float32_matmul_precision('highest' if precision=='bf16' else 'high')
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(precision not in ['bf16','bf16_tf32'])
        result=dict(precision=precision,backend=backend,batch=batch,status='running',gradient_checks=[],measurements=[])
        report.update(stage='candidate',active=dict(precision=precision,backend=backend,batch=batch));persist()
        try:
            apply_execution({'qwen':runtime},dict(attention_backend=backend,gradient_checkpointing=False))
            runtime.model.train();runtime.head.train();torch.cuda.reset_peak_memory_stats()
            result['bf16_drift']=[]
            for label in [0,1]:
                clear();features=fixtures[label][1][:4]
                chosen=selection([-3.,-3.,4.,4.],dict(threshold=0.,scale=1.))
                z=padded_forward(runtime.forward,features,4);loss,_=objective(z,label,chosen);loss.backward()
                actual=vector().cpu();expected=baseline[label]['gradient'];norm=float(expected.norm())
                rel=float((actual-expected).norm()/max(norm,1e-12))
                cos=float(torch.nn.functional.cosine_similarity(actual.double(),expected.double(),dim=0))
                result['bf16_drift'].append(dict(label=label,relative_l2=rel,cosine=cos,
                    logit_max_abs=float((z.detach().cpu()-baseline[label]['logits']).abs().max()),
                    finite=bool(torch.isfinite(actual).all()),passed=rel<=.1 and cos>=.99))
                clear();del actual,z,loss
            if not all(x['passed'] for x in result['bf16_drift']):
                result['status']='precision_drift_gate_failed';report['candidates'].append(result);persist();continue
            for label in [0,1]:
                features=fixtures[label][1][:batch]
                teacher=[-3.]*(len(features)//2)+[4.]*(len(features)-len(features)//2)
                chosen=selection(teacher,dict(threshold=0.,scale=1.))
                clear();z=padded_forward(runtime.forward,features,batch)
                reference,_=objective(z,label,chosen);reference.backward();expected=vector().clone()
                expected_loss=float(reference.detach());del reference,z
                clear();value,info=backward_fixed_selection(runtime.forward,features,label,chosen,batch=batch)
                actual=vector();norm=float(expected.norm())
                relative=float((actual-expected).norm()/max(norm,1e-12))
                cosine=float(torch.nn.functional.cosine_similarity(actual.double(),expected.double(),dim=0)) if norm else 1.
                result['gradient_checks'].append(dict(label=label,relative_l2=relative,cosine=cosine,
                    reference_loss=expected_loss,fast_loss=value,passed=relative<.01 and cosine>.999))
                del actual,expected;clear()
            if not all(c['passed'] for c in result['gradient_checks']):
                result['status']='gradient_gate_failed'
            else:
                for record,features,chosen in fixtures:
                    clear()
                    backward_fixed_selection(runtime.forward,features[:batch],record['label'],
                        selection(chosen['teacher'][:batch],dict(threshold=0.,scale=1.)),batch=batch)
                    clear();torch.cuda.synchronize();begin=time.time()
                    value,info=backward_fixed_selection(runtime.forward,features,record['label'],chosen,batch=batch)
                    torch.cuda.synchronize();seconds=time.time()-begin;clear()
                    result['measurements'].append(dict(name=record['name'],label=record['label'],windows=len(features),seconds=seconds,execution=info))
                result['training_hours_estimate']=sum(by_class[m['label']]*m['seconds']/m['windows'] for m in result['measurements'])/3600
                result['status']='passed'
            result['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
        except torch.OutOfMemoryError:
            result.update(status='out_of_memory');clear();torch.cuda.empty_cache()
        except Exception:
            result.update(status='error',error=traceback.format_exc());clear();torch.cuda.empty_cache()
        report['candidates'].append(result);persist();print('OPTIMIZATION_CANDIDATE',json.dumps(result),flush=True)
    eligible=[r for r in report['candidates'] if r['status']=='passed']
    report['best']=min(eligible,key=lambda r:r['training_hours_estimate']) if eligible else None
    report.update(status='benchmarked',stage='finished',elapsed_s=time.time()-report['started'],gpu=torch.cuda.get_device_name(),
        limitation='Initial two-fixture extrapolation only; full launch requires larger representative and end-to-end timing below 3h including validation/publication')
    persist();print('OPTIMIZATION_COMPLETE',flush=True)
except Exception:
    report.update(status='error',error=traceback.format_exc());persist();raise
