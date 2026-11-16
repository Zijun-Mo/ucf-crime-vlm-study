"""Record observed CU balance and a separate session-time estimate, including failures."""
import collections
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.cloud_io import sanitize
now=datetime.datetime.now(datetime.timezone.utc)
output=subprocess.run(['colab','usage'],capture_output=True,text=True,timeout=60,check=True).stdout
balance=re.search(r'Current balance:\s*([0-9.]+)',output)
active=re.search(r'Active assignments:\s*(\d+)',output)
assert balance and active,'Could not parse a fresh Colab account observation'
receipts=[json.loads(p.read_text()) for p in (ROOT/'research/runs').glob('*/receipt.json')]
receipts += json.loads((ROOT/'research/gpu_preflight_attempts.json').read_text())
rates={}
for receipt in receipts:
    match=re.search(r'Usage rate:\s*([0-9.]+)/hr',receipt.get('usage_allocated',''))
    gpu=receipt.get('gpu',receipt.get('requested_gpu'))
    if match and float(match.group(1))>0:
        rates[gpu]=float(match.group(1))
segments=[]
history=Path.home()/'.config/colab-cli/history'
for path in sorted(history.glob('ucf-*.jsonl')):
    opened=None
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row.get('event_type')=='session_created':
            if opened is not None:
                opened.update(termination_missing=True,latest_possible_end=row['timestamp'])
                segments.append(opened)
            opened={'session':path.stem,'endpoint':row['endpoint'],'accelerator':row['accelerator'],
                    'created_utc':row['timestamp'],'history':str(path)}
        elif row.get('event_type')=='session_terminated' and opened is not None:
            opened['terminated_utc']=row['timestamp'];segments.append(opened);opened=None
    if opened is not None:segments.append(opened)
for row in segments:
    end=row.get('terminated_utc')
    if end:
        seconds=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(row['created_utc'])).total_seconds()
        row.update(observed_session_seconds=seconds,interval_kind='CLI allocation-to-termination observations')
    else:
        seconds=None
        row['interval_kind']='No terminal history event; verify live assignment/receipt before interpreting time'
    rate=rates.get(row['accelerator'])
    row['rate_cu_hr']=rate
    row['rate_source']='observed allocation rate for same accelerator in this study' if rate is not None else 'unpriced; no verified rate in saved allocation receipts'
    row['estimated_cu']=seconds*rate/3600 if seconds is not None and rate is not None else None
initial=json.loads((ROOT/'research/protocol.json').read_text())['cost_policy']['initial_balance_cu']
report={'observed_utc':now.isoformat(),'initial_observed_balance_cu':initial,
        'latest_observed_balance_cu':float(balance.group(1)),
        'observed_unadjusted_balance_decrease_cu':initial-float(balance.group(1)),
        'observed_credit_grants_cu':json.loads((ROOT/'research/credit_refresh_audit.json').read_text())['observed_increment_cu'],
        'estimated_net_decrease_adjusted_for_observed_grant_cu':initial+json.loads((ROOT/'research/credit_refresh_audit.json').read_text())['observed_increment_cu']-float(balance.group(1)),
        'grant_evidence':'research/credit_refresh_audit.json',
        'active_assignments':int(active.group(1)),'usage_output':sanitize(output),'sessions':segments,
        'estimated_cu_closed_priced_sessions':sum(r['estimated_cu'] or 0 for r in segments),
        'unpriced_closed_sessions':[r['session'] for r in segments if r.get('terminated_utc') and r['rate_cu_hr'] is None],
        'unterminated_history_sessions':[r['session'] for r in segments if not r.get('terminated_utc')],
        'note':'The 200 CU observed balance increase is adjusted for; unadjusted balance decrease is not spending. Balance can lag. Time/rate estimates include failed attempts but exclude unpriced CPU intervals and intervals missing termination events. These estimates and balance changes are distinct observations, not exact debit accounting.'}
(ROOT/'research/cost_ledger.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['sessions','usage_output']},indent=2))
