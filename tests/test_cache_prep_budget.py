import json
from openjev_dss.followup_budget import spending, available


def test_cpu_cache_recovery_counts_against_same_new_training_envelope(tmp_path):
    p=tmp_path/'research/future_work_v1/budget.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(prior_phase_estimate_cu=68.,total_limit_cu=100.,limits_cu={'v3_training':20.,'recovery_reserve':3.4})))
    p=tmp_path/'research/localization_v4_plateau/cache_cpu/20261115/receipt.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(localization_policy='research/localization_v4/policy.json',allocated_utc='2026-11-15T00:00:00+00:00',release_observed_utc='2026-11-15T01:00:00+00:00',usage_allocated='Usage rate: 0.08/hr')))
    assert spending(tmp_path)['phases']['v3_training']==.08
    assert available(tmp_path,'v3_training')==19.92
