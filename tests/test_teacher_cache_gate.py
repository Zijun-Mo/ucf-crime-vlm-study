import json
import pytest
from openjev_dss.teacher_cache_gate import require_complete_teacher


def test_partial_cache_cannot_fall_back_to_gpu_inference(tmp_path):
    records=[dict(name=n,sha256=n,split='train') for n in ['a','b']]
    (tmp_path/'a.json').write_text('{}')
    with pytest.raises(ValueError,match='recomputation forbidden'):
        require_complete_teacher(records,tmp_path,{},'fp')


def test_complete_cache_requires_frozen_identity_and_finite_full_scores(tmp_path):
    records=[dict(name='a',sha256='sha',split='val')];meta={'a':dict(window_count=2)}
    data=dict(source_sha256='sha',fingerprint='fp',teacher='pinned_zero_shot_adapter_disabled',windows=2,logits=[-1.,2.])
    p=tmp_path/'a.json';p.write_text(json.dumps(data))
    assert require_complete_teacher(records,tmp_path,meta,'fp')['videos']==1
    for replacement in [dict(fingerprint='wrong'),dict(logits=[0.]),dict(logits=[0.,float('nan')])]:
        p.write_text(json.dumps({**data,**replacement}))
        with pytest.raises(ValueError,match='mismatch'):require_complete_teacher(records,tmp_path,meta,'fp')
