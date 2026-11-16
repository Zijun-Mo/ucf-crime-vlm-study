"""Wake the authorized current chat once, after experiment settlement; no LLM polling."""
import argparse,datetime,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.desktop_events import call_desktop
from openjev_dss.storyboard_notification import completion_prompt
parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path);parser.add_argument('--thread',required=True);args=parser.parse_args()
event=args.receipt.parent/'completion_event.json'
while True:
    receipt=json.loads(args.receipt.read_text())
    if receipt.get('finished_utc'):break
    time.sleep(20)
if event.exists() and json.loads(event.read_text()).get('handled_utc'):raise SystemExit(0)
record=dict(status=receipt['status'],receipt=str(args.receipt),result=str(args.receipt.parent/'result.json'),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),routine_model_calls=0)
event.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
for attempt in range(120):
    if event.exists() and json.loads(event.read_text()).get('handled_utc'):break
    try:
        response=call_desktop('read_thread',dict(threadId=args.thread,turnLimit=1,includeOutputs=False,maxOutputCharsPerItem=0),args.thread,'storyboard-idle')
        snapshot=json.loads(next(x['text'] for x in response['contentItems'] if x['type']=='inputText'))
        if any(t.get('status')=='inProgress' for t in snapshot.get('turns',[])):
            time.sleep(10);continue
        prompt=completion_prompt(receipt,event,args.receipt)
        result=call_desktop('send_message_to_thread',dict(threadId=args.thread,prompt=prompt),args.thread,'storyboard-'+args.receipt.parent.name)
        record.update(delivered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());event.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');break
    except (TimeoutError,EOFError):
        record['delivery']='ambiguous';event.write_text(json.dumps(record,indent=2)+'\n');break
    except Exception as e:
        record['last_delivery_error']=type(e).__name__;event.write_text(json.dumps(record,indent=2)+'\n');time.sleep(30)
