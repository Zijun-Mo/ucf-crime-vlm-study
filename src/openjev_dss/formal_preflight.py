"""Real gradient checks on training fixtures before the multi-epoch workload."""
import json
from pathlib import Path
import torch
from .losses import loss
from .training import two_pass as legacy_two_pass
from .formal_backward import two_pass


def gradient_check(runtime, bags, diagnostic_path=None, strict=True):
    runtime.model.train();runtime.head.train()
    params=[p for p in runtime.model.parameters() if p.requires_grad]+[p for p in runtime.head.parameters() if p.requires_grad]
    assert len(bags)==2 and all(len(w)==2 for w in bags)
    packed=torch.cat(bags)
    def clear():
        runtime.model.zero_grad(set_to_none=True);runtime.head.zero_grad(set_to_none=True)
    def vector():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
    def dense_once():
        clear();z=runtime.forward(packed).float();z.retain_grad()
        objective,_=loss(list(z.split([2,2])),[0,1],False);objective.backward()
        return vector(),z.detach().clone(),z.grad.detach().clone()
    dense,reference_scores,upstream=dense_once()
    def compare(leaf):
        assert torch.isfinite(dense).all() and torch.isfinite(leaf).all()
        norm=float(torch.linalg.vector_norm(dense))
        if norm:
            relative=float(torch.linalg.vector_norm(dense-leaf)/torch.linalg.vector_norm(dense))
            cosine=float(torch.nn.functional.cosine_similarity(dense.double(),leaf.double(),dim=0))
        else:relative=0.;cosine=1.;assert not leaf.any()
        return {'relative_l2':relative,'cosine':cosine,'dense_gradient_norm':norm}
    repeated,repeated_scores,repeated_upstream=dense_once()
    repeat=compare(repeated)
    repeat.update(score_max_abs=float((reference_scores-repeated_scores).abs().max()),
                  upstream_max_abs=float((upstream-repeated_upstream).abs().max()))
    clear();external_scores=runtime.forward(packed).float();external_scores.backward(upstream)
    external=compare(vector());external['score_max_abs']=float((reference_scores-external_scores.detach()).abs().max())
    def measure(implementation):
        clear();observed=[]
        options={'observer':lambda scores,g:observed.append((scores,g))} if implementation is two_pass else {}
        _,info=implementation(runtime.forward,bags,[0,1],False,chunk=4,score_batch=4,pad_batches=True,**options)
        result={**compare(vector()),'execution':info}
        if observed:
            scores,g=observed[0]
            result.update(score_max_abs=float((reference_scores-scores).abs().max()),upstream_max_abs=float((upstream-g).abs().max()))
        return result
    try:legacy=measure(legacy_two_pass)
    except Exception as error:legacy={'error':str(error)}
    report=measure(two_pass)
    report.update(passed=all(r['relative_l2']<.01 and r['cosine']>.999 for r in [report,repeat,external]),
                  physical_batch=4,train_only=True,legacy_comparison=legacy,dense_repeat=repeat,
                  direct_external_gradient=external,dense_logits=reference_scores.cpu().tolist(),upstream=upstream.cpu().tolist())
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(report,indent=2)+'\n')
    print('FORMAL_GRADIENT_CHECK',json.dumps(report),flush=True)
    clear()
    if strict:assert report['passed'],f"Formal dense/two-pass gradient mismatch: relative_l2={report['relative_l2']}, cosine={report['cosine']}, dense_repeat={repeat}"
    return report
