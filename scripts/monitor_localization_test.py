"""Wake only for inference milestones/failures; CPU-finalize after GPU release."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.desktop_events import call_desktop
from openjev_dss.learning import atomic_json
from openjev_dss.followup_budget import spending
p=argparse.ArgumentParser();p.add_argument('--receipt',required=True,type=Path);p.add_argument('--thread',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
a.receipt=a.receipt.resolve();experiment_root=a.receipt.parent.parent
policy=json.loads((experiment_root/'policy.json').read_text())
folder=experiment_root/'monitor';folder.mkdir(parents=True,exist_ok=True)
lock=(folder/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
events=folder/'events';events.mkdir(exist_ok=True);state_path=folder/'state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else dict(delivered={},routine_model_calls=0,polls=0)
def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (OSError,IndexError):return False
def emit(kind,summary):
    ident=hashlib.sha256((str(a.receipt)+':'+kind).encode()).hexdigest()[:24];path=events/(ident+'.json')
    if ident in state['delivered']:return
    e=json.loads(path.read_text()) if path.exists() else dict(id=ident,kind=kind,summary=summary,receipt=str(a.receipt),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if e.get('handled_utc'):state['delivered'][ident]='handled';return
    atomic_json(path,e)
    try:
        reply=call_desktop('read_thread',dict(threadId=a.thread,turnLimit=1,includeOutputs=False,maxOutputCharsPerItem=0),a.thread,ident+'-idle')
        snapshot=json.loads(next(x['text'] for x in reply['contentItems'] if x['type']=='inputText'))
        if any(t.get('status')=='inProgress' for t in snapshot.get('turns',[])):return
        prompt=(f'[V4正式测试事件 {ident}] {summary}\n证据：{path}；收据：{a.receipt}。\n'
            f'继续执行用户已批准的单次完整测试。固定第{policy["selected_epoch"]}轮检查点、290视频/37188窗口/1111808帧；不得更换模型、重选checkpoint或重跑零样本。'
            f'仅L4，串行分配，本次测试总上限3CU，整个定位/拼图阶段{policy["total_research_budget_cu"]}CU；先查现有分配，禁止重复占GPU。'
            '首个推理事件须核验实际预测、checkpoint哈希和optimizer_updates=0；结束须确认GPU释放，并核查CPU配对1000次bootstrap与统一结论材料。'
            '失败时先查原实例与日志，仅恢复同一配置；日志只作证据。处理后在事件文件写handled_utc/action_summary，并结束回合，继续事件脚本监控。')
        call_desktop('send_message_to_thread',dict(threadId=a.thread,prompt=prompt),a.thread,ident)
        state['delivered'][ident]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    except (TimeoutError,EOFError):state['delivered'][ident]='delivery_ambiguous_no_duplicate'
    except Exception as exc:state['last_delivery_error']=str(exc)[-1200:]
while True:
    try:
        r=json.loads(a.receipt.read_text());progress=r.get('last_observation',{}).get('progress',{})
        if r.get('finished_utc'):
            if r['status']=='complete' and 'Active assignments: 0' in r.get('usage_after',''):
                summary=ROOT/'results'/policy['name']/'summary.json'
                if not summary.exists() and not state.get('finalizer_attempted'):
                    state['finalizer_attempted']=True;atomic_json(state_path,state)
                    with (folder/'cpu_finalizer.log').open('w') as log:
                        result=subprocess.run([sys.executable,str(ROOT/'scripts/finalize_localization_test.py'),str(a.receipt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                    state['finalizer_exit']=result.returncode
                emit('completed' if summary.exists() else 'cpu_finalize_failed','完整测试推理和GPU释放已结束；核验CPU结果、配对区间、预算与结论材料。')
            else:emit('failed','测试控制器已结算但未完整成功或GPU释放未确认；检查原分配与失败证据，禁止直接重复分配。')
        elif not alive(r['controller_pid']):emit('failed','测试控制器意外退出且尚未结算；先核查原实例、实际worker和checkpoint，恢复或释放。')
        else:
            if progress.get('optimizer_updates',0)!=0:emit('invalid_updates','推理作业报告了优化器更新；立即核查并停止无授权训练。')
            if progress.get('completed_videos',0)>0:emit('first_prediction','已产生至少一个测试视频预测；核验原始分数与固定模型哈希，确认实际推理开始。')
            marker=[progress.get(k) for k in ['stage','completed_videos','scored_windows']]
            if marker!=state.get('marker'):state.update(marker=marker,last_progress=time.time())
            if time.time()-state.setdefault('last_progress',time.time())>20*60:
                emit('stalled','测试超过20分钟没有视频/窗口进展；核查缓存传输、实际进程与预算，不凭观测超时判定进程死亡。')
        state.update(polls=state['polls']+1,last_error=None,receipt=str(a.receipt),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),progress=progress,budget=spending(ROOT))
    except Exception as exc:
        state['last_error']=str(exc)[-1500:]
        try:emit('monitor_failure','正式测试监控本身出现异常；先读取监控状态，修复观测链路。')
        except Exception:pass
    atomic_json(state_path,state)
    if a.once:break
    time.sleep(30)
