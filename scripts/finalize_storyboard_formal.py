"""Verify and archive the registered test experiment, without GPU allocation."""
import datetime,hashlib,json,re
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/storyboard_context_formal_v1'
run=ROOT/'research/storyboard_context_formal/20261113T140017Z'
policy=json.loads((out/'policy.json').read_text())
result=json.loads((out/'result.json').read_text())
receipt=json.loads((run/'receipt.json').read_text())
worker=json.loads((run/'worker_status.json').read_text())
assert result==json.loads((run/'result.json').read_text())
assert result['status']==receipt['status']=='complete' and result['fingerprint']==policy['fingerprint']
assert worker['completed_videos']==290 and worker['scored_windows']==37188
assert result['optimizer_updates']==0 and receipt['stop_exit_code']==0
assert 'Active assignments: 0' in receipt['usage_after'] and 'Usage rate: 0.00/hr' in receipt['usage_after']
predictions={};checksums={};windows=0
for record in policy['records']:
    path=out/'predictions'/f'{record["name"]}.json';p=json.loads(path.read_text())
    n=policy['metadata'][record['name']]['window_count'];windows+=n
    assert p['name']==record['name'] and p['source_sha256']==record['sha256']
    assert p['fingerprint']==policy['fingerprint'] and p['start_s']==list(range(n))
    assert len(p['storyboard'])==len(p['baseline'])==n and 'prompt_only' not in p
    old=json.loads((ROOT/'results/controlled/qwen_zs_p1_4s/predictions'/path.name).read_text())
    assert p['baseline']==old['raw_logits'] and p['visual_cache_reused'] is True
    predictions[record['name']]=p;checksums[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
assert len(predictions)==len(list((out/'predictions').glob('*.json')))==290 and windows==37188
legacy=json.loads((ROOT/'results/study_formal_v2/qwen_zs_p1_4s.json').read_text())
for key,value in result['reports']['baseline']['metrics'].items():
    assert abs(value['value']-legacy['metrics'][key]['value'])<1e-12
gates=json.loads((out/'visual_cache_gates.json').read_text())
assert {r['image_tokens'] for r in gates}=={360,720}
assert all(r['allowed'] and r['max_abs_logit_error']==0. for r in gates)
rate=float(re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt['usage_allocated'])[1])
start=datetime.datetime.fromisoformat(receipt['allocated_utc']);end=datetime.datetime.fromisoformat(receipt['release_observed_utc'])
cost=(end-start).total_seconds()*rate/3600
prior=receipt['overall_budget_guard']['previous_experiments_estimate_cu']
assert cost<11 and prior+cost<40
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=dict(experiment='qwen_storyboard_context_formal_v1',status='complete',verified_utc=now,
    scope='Complete fixed test set: 290 videos, 1111808 frames, 37188 windows',
    inference_arms=['storyboard'],baseline_source=policy['baseline_source'],configuration_fingerprint=policy['fingerprint'],
    configuration_selected_on_test=False,primary_metric='pooled_frame_auroc',
    metrics={a:{k:v['value'] for k,v in r['metrics'].items()} for a,r in result['reports'].items()},
    paired_contrasts=result['paired_contrasts'],bootstrap=policy['bootstrap'],
    cache_audit=dict(baseline_logits_exactly_match_existing_all_290=True,visual_feature_hits=290,native_fallbacks=0,cache_gates=gates),
    timing=dict(allocation_hours=(end-start).total_seconds()/3600,worker_hours=result['allocation_worker_elapsed_s']/3600,
                completed_los_angeles=end.astimezone(ZoneInfo('America/Los_Angeles')).isoformat()),
    cost=dict(rate_time_estimate_cu=cost,quoted_rate_cu_per_hour=rate,
              prior_experiments_estimate_cu=prior,total_research_estimate_cu=prior+cost,total_research_budget_cu=40,
              limitation='Rate times observed allocation duration, not an itemized billing invoice'),
    gpu_released=True,optimizer_updates=0,
    conclusion='Pooled frame AUROC and within-video AUROC are essentially unchanged; paired confidence intervals include zero. Pooled frame AP decreases and its nominal paired 95% interval excludes zero. The development-pilot gains did not persist at full-test scale.',
    notification_recovery=dict(error='TypeError from an unintended unary plus on a string in prompt assembly',
        fixed=True,regression_checks='4 payload-construction cases passed; no messages sent by tests'))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(out/'prediction_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
(out/'allocation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
registry_path=ROOT/'research/formal_experiment_registry.json';registry=json.loads(registry_path.read_text())
registry['experiments']['qwen_storyboard_context_formal_v1'].update(status='complete',summary='results/storyboard_context_formal_v1/summary.json',
    verified_utc=now,gpu_released=True,completed_videos=290,scored_windows=37188)
registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
event_path=run/'completion_event.json';event=json.loads(event_path.read_text())
event.update(handled_utc=now,action_summary='Verified full coverage, exact reused baseline, 290 visual-cache hits, zero-score-error gates for both image token shapes, and zero active assignments. Archived all 290 predictions and formal summary. Primary AUROC +0.263218 pp (95% CI -3.587286 to +4.071211 pp); pooled AP -5.425632 pp (CI -9.898536 to -0.456721 pp). Fixed notifier unary-plus string defect and verified four regression cases. No new GPU or training launched.',formal_summary='results/storyboard_context_formal_v1/summary.json')
event_path.write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='complete',videos=290,frames=1111808,windows=37188,cost_estimate_cu=cost,total_research_estimate_cu=prior+cost,gpu_released=True),ensure_ascii=False))
