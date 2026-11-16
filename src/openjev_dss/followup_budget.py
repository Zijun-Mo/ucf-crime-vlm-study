"""Cost guards for the explicitly budgeted localization/storyboard phase."""
import datetime,json,re
from pathlib import Path


def amount(receipt,now=None):
    if not receipt.get('allocated_utc'):return 0.
    quote=receipt.get('usage_allocated',receipt.get('usage_at_allocation',''))
    rate=re.search(r'Usage rate:\s*([0-9.]+)/hr',quote)
    if not rate:raise ValueError('Unpriced follow-up allocation')
    start=datetime.datetime.fromisoformat(receipt['allocated_utc'])
    stop=receipt.get('release_observed_utc')
    if 'Active assignments:' in receipt.get('usage_after','') and 'Active assignments: 0' not in receipt['usage_after']:
        stop=None # A failed stop request is not evidence that billing ended.
    if not stop and 'Active assignments: 0' in receipt.get('usage_after',''):stop=receipt.get('finished_utc')
    end=datetime.datetime.fromisoformat(stop) if stop else (now or datetime.datetime.now(datetime.timezone.utc))
    return max(0.,(end-start).total_seconds())*float(rate.group(1))/3600


def spending(root):
    root=Path(root);budget=json.loads((root/'research/future_work_v1/budget.json').read_text())
    phases=dict(teacher_test=0.,v3_training=0.,v3_test=0.,annotation_cpu=0.)
    seen=set()
    for key,pattern in [('teacher_test','research/localization_v2_test/20*/receipt.json'),
                        ('v3_test','research/localization_v3_test/20*/receipt.json'),
                        ('v3_test','research/localization_v4_test/20*/receipt.json'),
                        ('v3_test','research/localization_v4_epoch5_test/20*/receipt.json'),
                        ('v3_test','research/localization_v4_plateau_test/20*/receipt.json'),
                        ('annotation_cpu','research/localization_dev_v2/allocations/*/receipt.json'),
                        ('annotation_cpu','research/localization_v4_test/analysis_cpu/*/receipt.json'),
                        ('annotation_cpu','research/localization_v4_epoch5_test/analysis_cpu/*/receipt.json'),
                        ('annotation_cpu','research/localization_v4_plateau_test/analysis_cpu/*/receipt.json'),
                        ('v3_training','research/localization_v4_plateau/cache_cpu/*/receipt.json'),
                        ('v3_training','research/runs/localization_train_*/receipt.json')]:
        for path in root.glob(pattern):
            if path.resolve() in seen:continue
            value=json.loads(path.read_text())
            if key=='v3_training' and value.get('localization_policy') not in ['research/localization_v3/policy.json','research/localization_v4/policy.json']:continue
            seen.add(path.resolve());phases[key]+=amount(value)
    total=budget['prior_phase_estimate_cu']+sum(phases.values())
    return dict(phases=phases,phase_prior_cu=budget['prior_phase_estimate_cu'],total_estimate_cu=total,
        remaining_cu=budget['total_limit_cu']-total,total_limit_cu=budget['total_limit_cu'],
        limitation='Quoted rate times observed runtime; concurrent annotation preparation is included in its GPU allocation, not charged twice')


def available(root,phase):
    root=Path(root);budget=json.loads((root/'research/future_work_v1/budget.json').read_text());observed=spending(root)
    return max(0.,min(budget['limits_cu'][phase]-observed['phases'][phase],observed['remaining_cu']-budget['limits_cu']['recovery_reserve']))
