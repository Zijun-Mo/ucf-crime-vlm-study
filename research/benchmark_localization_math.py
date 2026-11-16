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

out=Path('/content/ucf_localization_optimization_math');out.mkdir(exist_ok=True)
report=dict(status='running',stage='load',started=time.time(),candidates=[],full_training_started=False,test_used=False)
def persist():atomic_json(out/'status.json',report)
persist();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
try:
    base=snapshot_download('Qwen/Qwen3.5-2B',revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        local_files_only=True,allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'])
    protocol=json.loads((work/'research/protocol.json').read_text())
    runtime=ControlledRuntime(base,protocol['prompts']['P1'],train=True)
    manifest=json.loads(Path('/content/ucf_drive/weak_dss_learning_10cu_v1/manifest.json').read_text())
    records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    provider=FormalVisualProvider(runtime,'/content/ucf_formal_data',records,'/content/ucf_formal_visual',
        '15852e8c16360a2fea060d615a32b45270f8a8fc')
    params=[p for p in runtime.model.parameters() if p.requires_grad]
    def clear():runtime.model.zero_grad(set_to_none=True)
    def vector():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    fixtures=[]
    for label in [0,1]:
        record=min((r for r in records if r['split']=='train' and r['label']==label and 128<=provider.metadata[r['name']]['window_count']<=512),
            key=lambda r:(provider.metadata[r['name']]['window_count'],r['name']))
        view=provider(record,0,True);features=view[:min(128,len(view))]
        runtime.model.eval();runtime.head.eval()
        with runtime.model.disable_adapter():teacher=stream_scores(runtime.forward,features).tolist()
        chosen=selection(teacher,dict(threshold=float(np.median(teacher)),scale=1.))
        fixtures.append((record,features,chosen))
    by_class={y:sum(provider.metadata[r['name']]['window_count'] for r in records if r['split']=='train' and r['label']==y) for y in [0,1]}
    for backend,batch in [('math',8),('math',12),('math',16),('math',20)]:
        result=dict(backend=backend,batch=batch,status='running',gradient_checks=[],measurements=[])
        report.update(stage='candidate',active=dict(backend=backend,batch=batch));persist()
        try:
            apply_execution({'qwen':runtime},dict(attention_backend=backend,gradient_checkpointing=False))
            runtime.model.train();runtime.head.train();torch.cuda.reset_peak_memory_stats()
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
