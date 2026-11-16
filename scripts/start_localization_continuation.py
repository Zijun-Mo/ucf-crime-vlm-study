"""Start the authorized V4 continuation once, with its supervisor/event monitor."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',default='research/localization_v4_continuation/control.json')
    args=parser.parse_args()
    control=args.control;configuration=json.loads((ROOT/control).read_text())
    assert configuration['authorized'] is True
    policy=configuration['base_policy'];folder=(ROOT/control).parent
    previous_path=folder/'launch.json'
    if not previous_path.exists():previous_path=ROOT/configuration['previous_launch']
    previous=json.loads(previous_path.read_text())
    old=json.loads((ROOT/previous['receipt']).read_text())
    assert old.get('finished_utc') and 'Active assignments: 0' in old.get('usage_after',''), 'Existing job not settled; do not allocate'
    for pid in [previous['controller_pid'],previous['supervisor_pid'],previous['monitor_pid']]:
        stat=Path('/proc')/str(pid)/'stat'
        assert not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]=='Z','Previous controller/supervisor still active'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (folder/('launch_before_'+stamp+'.json')).write_text(json.dumps(previous,indent=2)+'\n')
    # Event handling belongs to the caller after inspecting the outcome.
    # Launching alone is not evidence that a recovery event has succeeded.
    before=set((ROOT/'research/runs').glob('localization_train_*/receipt.json'))
    cmd=[sys.executable,str(ROOT/'scripts/launch_study_job.py'),'localization_train','--gpu','L4','--minutes','460','--localization-phase','train','--localization-policy',policy,'--continuation',control]
    env={**os.environ,'PYTHONUNBUFFERED':'1'}
    def start(command,log):
        with (folder/log).open('a') as out:
            return subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    controller=start(cmd,'controller.log');receipt=None
    for _ in range(50):
        fresh=set((ROOT/'research/runs').glob('localization_train_*/receipt.json'))-before
        if len(fresh)==1:receipt=fresh.pop();break
        assert controller.poll() is None,'Controller exited before receipt'
        time.sleep(.2)
    assert receipt,'Inspect controller;no duplicate start'
    supervisor=start([sys.executable,str(ROOT/'scripts/continue_localization_training.py'),'--current-receipt',str(receipt),'--allocations','5'],'supervisor.log')
    monitor=start([sys.executable,str(ROOT/'scripts/monitor_localization_events.py'),'--thread',previous['thread'],'--policy',policy,'--continuation',control,'--interval','60'],'monitor.log')
    state=dict(status='preparing',launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=controller.pid,supervisor_pid=supervisor.pid,monitor_pid=monitor.pid,command=cmd,receipt=str(receipt.relative_to(ROOT)),continuation_control=control,thread=previous['thread'],resume_steps=configuration['resume_steps'],training_started=False,actual_new_update_verified=False)
    (folder/'launch.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':main()
