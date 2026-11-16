"""Freeze a training-only execution path after both arms pass the same AD gate."""
import json
from pathlib import Path
import time
import torch
from .formal_preflight import gradient_check


def apply_execution(runtimes, choice):
    assert choice in [{'attention_backend':'auto','gradient_checkpointing':False},
                      {'attention_backend':'math','gradient_checkpointing':False}]
    for runtime in runtimes.values():
        runtime.model.gradient_checkpointing_disable()
        if not hasattr(runtime,'_formal_original_forward'):runtime._formal_original_forward=runtime.forward
        base=runtime._formal_original_forward
        def forward(visual,runtime=runtime,base=base,backend=choice['attention_backend']):
            if runtime.model.training and backend=='math':
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with sdpa_kernel(SDPBackend.MATH):return base(visual)
            # Validation uses the original AUTO inference execution path.
            return base(visual)
        runtime.forward=forward


def select_execution(runtimes,bags,out):
    assert set(runtimes)=={'qwen','openjev'}
    out=Path(out);frozen=out/'training_execution.json'
    choices=([json.loads(frozen.read_text())['choice']] if frozen.exists() else
             [{'attention_backend':'auto','gradient_checkpointing':False},
              {'attention_backend':'math','gradient_checkpointing':False}])
    attempts=[]
    for choice in choices:
        apply_execution(runtimes,choice);results={};started=time.time()
        cuda=torch.cuda.is_available() and any(str(r.device).startswith('cuda') for r in runtimes.values())
        if cuda:torch.cuda.reset_peak_memory_stats()
        for name,runtime in runtimes.items():
            try:
                results[name]=gradient_check(runtime,bags,out/(choice['attention_backend']+'_'+name+'_gradient_diagnostic.json'),strict=False)
            except Exception as error:
                results[name]={'passed':False,'error':str(error)}
            finally:
                runtime.model.zero_grad(set_to_none=True);runtime.head.zero_grad(set_to_none=True)
        peak=torch.cuda.max_memory_allocated()/1024**3 if cuda else None
        headroom=torch.cuda.get_device_properties(0).total_memory/1024**3-peak if cuda else None
        attempts.append({'choice':choice,'results':results,'seconds':time.time()-started,
                         'peak_gpu_gib':peak,'optimizer_headroom_gib':headroom})
        (out/'execution_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
        if all(r['passed'] for r in results.values()) and (headroom is None or headroom>=1.):
            if not frozen.exists():
                frozen.write_text(json.dumps({'choice':choice,'selected_before_optimizer_updates':True,'train_only':True,
                    'selection':'First common execution path passing unchanged relative-L2/cosine gates for dense repeat, direct external gradient, and aligned two-pass on both training fixtures',
                    'validation_attention_backend':'original AUTO; identical to frozen test producer'},indent=2)+'\n')
            return choice,results
    raise RuntimeError('No common formal execution path passed the unchanged dense gradient/stability gate; no optimizer updates allowed')
