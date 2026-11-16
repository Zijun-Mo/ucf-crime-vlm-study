"""Experimental train-only FP8 frozen GEMMs; native BF16 teacher/eval preserved."""
import types
import torch
import torch.nn.functional as F


def quantize_tensor(tensor,dtype=torch.float8_e4m3fn):
    bound=torch.finfo(dtype).max
    scale=tensor.detach().float().abs().amax().clamp_min(1e-12)/bound
    value=(tensor.float()/scale).clamp(-bound,bound).to(dtype)
    return value,scale


def scaled(a,b,sa,sb):
    rows=a.shape[0];padding=(-rows)%16
    if padding:a=F.pad(a,(0,0,0,padding))
    value=torch._scaled_mm(a,b,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16,use_fast_accum=False)
    return value[:rows]


class FrozenFloat8(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs,weight8,weight8_transpose,weight_scale,weight_bf16,backward_fp8):
        ctx.shape=inputs.shape;ctx.backward_fp8=backward_fp8
        ctx.save_for_backward(weight8_transpose,weight_scale,weight_bf16)
        x,sx=quantize_tensor(inputs.reshape(-1,inputs.shape[-1]))
        result=scaled(x,weight8.t(),sx,weight_scale)
        return result.reshape(*inputs.shape[:-1],weight8.shape[0])

    @staticmethod
    def backward(ctx,grad_output):
        transposed,scale,original=ctx.saved_tensors
        gradient=grad_output.reshape(-1,grad_output.shape[-1]).contiguous()
        if ctx.backward_fp8:
            g,sg=quantize_tensor(gradient,torch.float8_e5m2)
            result=scaled(g,transposed.t(),sg,scale)
        else:result=gradient.to(original.dtype)@original
        return result.reshape(ctx.shape),None,None,None,None,None


def patch_frozen_linears(runtime):
    """Quantize frozen language projections only; keep original weights intact."""
    modules=[]
    for name,module in runtime.model.named_modules():
        if type(module) is not torch.nn.Linear or module.weight.requires_grad:continue
        if module.bias is not None or module.in_features%16 or module.out_features%16:continue
        original=module.forward
        with torch.no_grad():
            w,s=quantize_tensor(module.weight)
            wt=w.t().contiguous()
        module.register_buffer('_localization_fp8_weight',w,persistent=False)
        module.register_buffer('_localization_fp8_weight_t',wt,persistent=False)
        module.register_buffer('_localization_fp8_scale',s,persistent=False)
        module._localization_precision='bf16'
        def forward(self,x,_original=original):
            if not self.training or self._localization_precision=='bf16':return _original(x)
            return FrozenFloat8.apply(x,self._localization_fp8_weight,self._localization_fp8_weight_t,
                self._localization_fp8_scale,self.weight,self._localization_precision=='fp8_full')
        module.forward=types.MethodType(forward,module);modules.append((name,module))
    assert modules,'No frozen projections eligible for FP8'
    return modules


def set_precision(modules,precision):
    assert precision in ['bf16','fp8_forward','fp8_full']
    for _,module in modules:module._localization_precision=precision
