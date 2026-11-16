from openjev_dss.trial_events import detect_events


def run(progress,receipt=None,supervisor=None,online=None,now=0,state=None):
    return detect_events('receipt',receipt or {'status':'running','allocated_utc':'now'},
        supervisor or {'status':'running'},progress,online or {'fresh_progress':True},now,state if state is not None else {})


def test_normal_progress_never_triggers_agent_polling():
    state={}
    for i in range(100):assert not run(dict(status='running',stage='teacher_scoring',teacher_videos=i),now=i*60,state=state)


def test_first_update_and_validation_have_stable_dedup_ids():
    p=dict(status='running',stage='training',steps=1,epoch_cursor=0,video_cursor=8)
    a=run(p);b=run({**p,'steps':2,'video_cursor':16})
    assert a[0]['kind']=='first_update' and a[0]['id']==b[0]['id']
    a=run({**p,'validation':[dict(epoch=1)]});b=run({**p,'validation':[dict(epoch=2)]})
    assert a[-1]['kind']=='validation' and a[-1]['id']!=b[-1]['id']


def test_failure_and_completion_are_necessary_single_events():
    a=run({},receipt=dict(status='incomplete',finished_utc='now'))
    assert len(a)==1 and a[0]['kind']=='failure'
    a=run(dict(status='complete',steps=471,epoch_cursor=3,validation=[dict(epoch=3)]))
    assert len(a)==1 and a[0]['kind']=='completed'


def test_stage_specific_stall_and_recovery_progress():
    state={};p=dict(status='running',stage='training',steps=2,epoch_cursor=0)
    run(p,now=0,state=state)
    assert any(x['kind']=='stalled' for x in run(p,now=13*60,state=state))
    assert not any(x['kind']=='stalled' for x in run({**p,'steps':3},now=14*60,state=state))
    state={};p=dict(status='running',stage='validation',steps=157,epoch_cursor=0)
    run(p,now=0,state=state)
    assert not any(x['kind']=='stalled' for x in run(p,now=40*60,state=state))


def test_single_network_miss_does_not_wake_but_persistent_blindness_does():
    state={};p=dict(status='running',stage='training',steps=3)
    run(p,now=0,state=state)
    assert not any(x['kind']=='monitor_blind' for x in run(p,online={'observation_error':'SSLError'},now=60,state=state))
    assert any(x['kind']=='monitor_blind' for x in run(p,online={'observation_error':'SSLError'},now=16*60,state=state))


def test_budget_continuation_is_not_a_training_failure():
    receipt=dict(status='incomplete',finished_utc='now',training_status='budget_stopped')
    assert not run(dict(status='budget_stopped'),receipt=receipt)


def test_preparation_budget_alert_does_not_reset_on_teacher_progress():
    from openjev_dss.trial_events import detect_events
    import datetime
    start=datetime.datetime.fromisoformat('2026-11-14T00:00:00+00:00').timestamp()
    receipt=dict(allocated_utc='2026-11-14T00:00:00+00:00',startup_without_update_limit_minutes=45)
    def events(minutes,steps):
        return detect_events('receipt',receipt,{},dict(status='running',stage='teacher_scoring',steps=steps,teacher_videos=940),{'fresh_progress':True},start+minutes*60,{})
    assert not any(e['kind']=='preparation_limit' for e in events(44,0))
    assert any(e['kind']=='preparation_limit' for e in events(46,0))
    assert not any(e['kind']=='preparation_limit' for e in events(46,1))


def test_continuation_does_not_report_old_steps_or_validation_as_new():
    receipt=dict(status='running',resume_steps=328,validation_epoch_floor=2,event_scope='extension')
    old=dict(status='running',stage='training',steps=328,validation=[dict(epoch=2)])
    assert not run(old,receipt=receipt)
    new=run({**old,'steps':329},receipt=receipt)
    assert [r['kind'] for r in new]==['first_update']
    original=run(dict(status='running',steps=1))[0]
    assert original['id']!=new[0]['id']
    latest=run({**old,'steps':471,'validation':[dict(epoch=3)]},receipt=receipt)
    assert [r['kind'] for r in latest]==['first_update','validation']


def test_exhausted_continuation_budget_is_a_necessary_stop_event():
    result=run(dict(status='budget_stopped'),supervisor={'status':'budget_exhausted'})
    assert len(result)==1 and result[0]['kind']=='budget_limit'


def test_inherited_terminal_pointer_during_copy_is_not_new_completion():
    receipt=dict(status='running',resume_steps=328,validation_epoch_floor=2,event_scope='extension')
    old=dict(status='budget_complete',stage='finished',steps=328,validation=[dict(epoch=2)])
    assert not run(old,receipt=receipt)
    completed=run({**old,'status':'complete','steps':471,'epoch_cursor':3},receipt=receipt)
    assert len(completed)==1 and completed[0]['kind']=='completed'


def test_inherited_user_stop_is_asset_restore_not_a_twenty_minute_finished_stage_stall():
    receipt=dict(status='running',resume_steps=806,validation_epoch_floor=5,event_scope='plateau')
    old=dict(status='user_stopped',stage='finished',steps=806,validation=[dict(epoch=5)])
    state={}
    assert not run(old,receipt=receipt,state=state,now=0)
    assert not run(old,receipt=receipt,state=state,now=21*60)
    assert state['marker'][1]=='restoring_assets'
    assert [e['kind'] for e in run({**old,'status':'running','steps':807},receipt=receipt)]==['first_update']


def test_missing_ssl_read_after_confirmed_updates_is_not_fresh_startup_or_setup_stall():
    import datetime
    start=datetime.datetime.fromisoformat('2026-11-15T00:00:00+00:00').timestamp()
    receipt=dict(status='running',allocated_utc='2026-11-15T00:00:00+00:00',resume_steps=806,startup_without_update_limit_minutes=45)
    state={};progress=dict(status='running',stage='validation',steps=942,video_cursor=1256)
    run(progress,receipt=receipt,state=state,now=start+90*60)
    marker=state['marker'][:];last_progress=state['last_progress']
    missed=run({},receipt=receipt,state=state,online={'progress_http_status':'SSLError','worker_alive':True},now=start+91*60)
    assert not missed
    assert state['max_observed_steps']==942 and state['marker']==marker and state['last_progress']==last_progress
    prolonged=run({},receipt=receipt,state=state,online={'observation_error':'Unavailable'},now=start+106*60)
    assert [e['kind'] for e in prolonged]==['monitor_blind']


def test_persisted_verified_update_prevents_false_startup_after_monitor_restart():
    import datetime
    start=datetime.datetime.fromisoformat('2026-11-15T00:00:00+00:00').timestamp()
    receipt=dict(status='running',allocated_utc='2026-11-15T00:00:00+00:00',resume_steps=806,startup_without_update_limit_minutes=45,verified_optimizer_steps=807)
    assert not run({},receipt=receipt,state={},online={'observation_error':'Unavailable'},now=start+100*60)


def test_receipt_change_does_not_carry_another_allocations_update_evidence():
    import datetime
    start=datetime.datetime.fromisoformat('2026-11-15T00:00:00+00:00').timestamp()
    state={};run(dict(status='running',stage='training',steps=950),state=state,now=start)
    receipt=dict(status='running',allocated_utc='2026-11-15T00:00:00+00:00',resume_steps=806,startup_without_update_limit_minutes=45)
    events=detect_events('different_receipt',receipt,{}, {},{'observation_error':'Unavailable'},start+46*60,state)
    assert any(e['kind']=='preparation_limit' for e in events)
