import copy
import pytest
from openjev_dss.continuation_control import verify_resume


def test_optimizer_cursor_must_be_preserved_and_later_allocations_can_advance():
    control=dict(base_fingerprint='fp',resume_steps=328,completed_epochs=2,resume_video_cursor=112)
    state=dict(fingerprint='fp',steps=328,epoch_cursor=2,video_cursor=112)
    saved=dict(fingerprint='fp',steps=328,optimizer={'state':{0:{'step':328},1:{'step':328}}})
    assert verify_resume(control,state,saved)==[328,328]
    bad=copy.deepcopy(saved);bad['optimizer']['state'][0]['step']=0
    with pytest.raises(AssertionError):verify_resume(control,state,bad)
    with pytest.raises(AssertionError):verify_resume(control,{**state,'video_cursor':0},saved)
    later=copy.deepcopy(saved);later['steps']=471
    for row in later['optimizer']['state'].values():row['step']=471
    assert verify_resume(control,{**state,'steps':471,'epoch_cursor':3,'video_cursor':0},later)==[471,471]


def test_resume_cannot_reset_trajectory_or_reuse_other_protocol():
    control=dict(base_fingerprint='fp',resume_steps=328,completed_epochs=2,resume_video_cursor=112)
    state=dict(fingerprint='fp',steps=0,epoch_cursor=0,video_cursor=0)
    saved=dict(fingerprint='other',steps=0,optimizer={'state':{}})
    with pytest.raises(AssertionError):verify_resume(control,state,saved)


def test_cache_recovery_cannot_allocate_training_before_bundle_is_verified(tmp_path):
    import hashlib,json
    from openjev_dss.continuation_control import load_continuation
    policy=tmp_path/'policy.json';policy.write_text('{}')
    control=dict(authorized=True,base_policy='policy.json',base_policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        until_validation_plateau=True,additional_training_limit_cu=20.,gpu_allowlist=['L4'],test_during_training=False,
        resume_remote='old',output_remote='new',resume_requires_cache_bundle=True)
    path=tmp_path/'control.json';path.write_text(json.dumps(control))
    with pytest.raises(AssertionError,match='Verified CPU cache bundle'):load_continuation(tmp_path,'control.json','policy.json')
    control['visual_cache_bundle']=dict(manifest_sha256='pinned',inventory_sha256='pinned')
    path.write_text(json.dumps(control))
    assert load_continuation(tmp_path,'control.json','policy.json')['authorized']
