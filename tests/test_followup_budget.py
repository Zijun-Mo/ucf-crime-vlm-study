import datetime,json
from openjev_dss.followup_budget import amount,spending,available


def test_unknown_release_keeps_accruing_after_controller_exit():
    row=dict(allocated_utc='2026-11-14T00:00:00+00:00',finished_utc='2026-11-14T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr',usage_after='Active assignments: 1')
    assert amount(row,datetime.datetime.fromisoformat('2026-11-14T02:00:00+00:00'))==3.88


def test_phase_cap_excludes_historical_training_and_does_not_double_count_cpu(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    write('research/future_work_v1/budget.json',dict(prior_phase_estimate_cu=35.,total_limit_cu=60.,limits_cu={'v3_training':15.,'teacher_test':3.,'recovery_reserve':3.4}))
    receipt=dict(allocated_utc='2026-11-14T00:00:00+00:00',release_observed_utc='2026-11-14T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr')
    write('research/localization_v2_test/20261114/receipt.json',receipt)
    write('research/runs/localization_train_new/receipt.json',{**receipt,'localization_policy':'research/localization_v3/policy.json'})
    write('research/runs/localization_train_old/receipt.json',{**receipt,'localization_policy':'research/localization_v2/policy.json'})
    value=spending(tmp_path)
    assert value['phases']['v3_training']==1.94 and value['phases']['teacher_test']==1.94
    assert value['phases']['annotation_cpu']==0
    assert abs(value['total_estimate_cu']-38.88)<1e-10
    assert abs(available(tmp_path,'v3_training')-13.06)<1e-10


def test_v4_uses_same_fifteen_cu_training_envelope(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    write('research/future_work_v1/budget.json',dict(prior_phase_estimate_cu=35.,total_limit_cu=60.,limits_cu={'v3_training':15.,'recovery_reserve':3.4}))
    receipt=dict(allocated_utc='2026-11-14T00:00:00+00:00',release_observed_utc='2026-11-14T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr')
    for version in [3,4]:write(f'research/runs/localization_train_v{version}/receipt.json',{**receipt,'localization_policy':f'research/localization_v{version}/policy.json'})
    assert spending(tmp_path)['phases']['v3_training']==3.88
    assert abs(available(tmp_path,'v3_training')-11.12)<1e-10


def test_stop_timestamp_with_active_assignment_does_not_stop_billing():
    row=dict(allocated_utc='2026-11-14T00:00:00+00:00',release_observed_utc='2026-11-14T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr',usage_after='Active assignments: 1')
    assert amount(row,datetime.datetime.fromisoformat('2026-11-14T02:00:00+00:00'))==3.88


def test_v4_test_uses_same_three_cu_test_envelope(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    write('research/future_work_v1/budget.json',dict(prior_phase_estimate_cu=35.,total_limit_cu=60.,limits_cu={'v3_test':3.,'recovery_reserve':3.4}))
    r=dict(allocated_utc='2026-11-14T00:00:00+00:00',release_observed_utc='2026-11-14T00:30:00+00:00',usage_allocated='Usage rate: 1.94/hr')
    write('research/localization_v4_test/20261114/receipt.json',r)
    assert spending(tmp_path)['phases']['v3_test']==.97
    assert abs(available(tmp_path,'v3_test')-2.03)<1e-10


def test_explicit_epoch5_test_is_counted_in_the_amended_test_envelope(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    write('research/future_work_v1/budget.json',dict(prior_phase_estimate_cu=35.,total_limit_cu=80.,limits_cu={'v3_test':5.,'recovery_reserve':3.4}))
    r=dict(allocated_utc='2026-11-15T00:00:00+00:00',release_observed_utc='2026-11-15T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr')
    write('research/localization_v4_test/20261114/receipt.json',r)
    write('research/localization_v4_epoch5_test/20261115/receipt.json',r)
    assert spending(tmp_path)['phases']['v3_test']==3.88
    assert abs(available(tmp_path,'v3_test')-1.12)<1e-10


def test_plateau_selected_test_is_counted_alongside_preserved_tests(tmp_path):
    budget=tmp_path/'research/future_work_v1/budget.json'
    budget.parent.mkdir(parents=True)
    budget.write_text(json.dumps(dict(prior_phase_estimate_cu=68.,total_limit_cu=100.,limits_cu={'v3_test':6.,'recovery_reserve':3.4})))
    receipt=dict(allocated_utc='2026-11-15T00:00:00+00:00',release_observed_utc='2026-11-15T01:00:00+00:00',usage_allocated='Usage rate: 1.94/hr')
    for folder in ['localization_v4_epoch5_test','localization_v4_plateau_test']:
        p=tmp_path/f'research/{folder}/20261115/receipt.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(receipt))
    p=tmp_path/'research/localization_v4_plateau_test/analysis_cpu/20261115/receipt.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({**receipt,'usage_allocated':'Usage rate: 0.08/hr'}))
    assert spending(tmp_path)['phases']['v3_test']==3.88
    assert spending(tmp_path)['phases']['annotation_cpu']==.08
    assert abs(available(tmp_path,'v3_test')-2.12)<1e-10
