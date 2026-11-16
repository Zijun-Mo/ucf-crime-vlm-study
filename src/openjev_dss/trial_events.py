"""Pure event detection: polling itself never invokes a language model."""
import hashlib
import datetime


TERMINAL={'complete','budget_complete','speed_gate_stopped'}


def detect_events(receipt_path,receipt,supervisor,progress,online,now,state,controller_alive=True):
    if state.get('receipt_path')!=receipt_path:
        state.update(receipt_path=receipt_path,max_observed_steps=0,missing_assignment=0)
        for key in ['marker','last_progress','last_contact']:state.pop(key,None)
    observed=progress.get('steps')
    verified=receipt.get('verified_optimizer_steps',0)
    state['max_observed_steps']=max(state.get('max_observed_steps',0),
        observed if isinstance(observed,int) else 0,verified)
    def event(kind,summary,tag='',global_key=False):
        scope=receipt.get('event_scope','')
        key=(f'{scope}:' if scope else '')+(f'{kind}:{tag}' if global_key else f'{receipt_path}:{kind}:{tag}')
        return dict(id=hashlib.sha256(key.encode()).hexdigest()[:24],kind=kind,summary=summary,
                    receipt=receipt_path,progress=progress)
    status=progress.get('status')
    if (receipt.get('resume_steps') and not receipt.get('finished_utc')
            and progress.get('steps',0)<=receipt['resume_steps'] and status in TERMINAL|{'user_stopped'}):
        # A copied status pointer may arrive before its optimizer payload.
        # This is the original run's terminal marker, not the new run finishing.
        status='running'
        progress={**progress,'status':'running','stage':'restoring_assets'}
    if supervisor.get('status')=='budget_exhausted':
        return [event('budget_limit','续训预算不足以覆盖下一完整轮次，已停止新增分配；尚不能称为收敛。',global_key=True)]
    if status in TERMINAL:
        return [event('completed' if status!='speed_gate_stopped' else 'speed_limit',
            f"作业状态 {status}；更新 {progress.get('steps',0)} 步，完成 {progress.get('epoch_cursor',0)} 轮。",global_key=True)]
    result_status=receipt.get('training_status')
    expected_continuation=result_status in ['budget_stopped','prepared']
    failed=(receipt.get('status')=='incomplete' and bool(receipt.get('finished_utc')) and not expected_continuation)
    failed|=supervisor.get('status') in ['needs_inspection','allocation_limit_reached']
    failed|=not receipt.get('finished_utc') and not controller_alive
    if failed:return [event('failure','训练控制器或作业已异常退出；核查最新失败日志并修复恢复。')]
    if online.get('assignment_present') is False and receipt.get('allocated_utc') and not receipt.get('finished_utc'):
        state['missing_assignment']=state.get('missing_assignment',0)+1
        if state['missing_assignment']>=2:return [event('failure','远端训练实例连续两次确认已消失。')]
    else:state['missing_assignment']=0
    if online.get('worker_alive') is False and not receipt.get('finished_utc'):
        return [event('failure','远端训练进程已退出；先检查原因和已提交检查点。')]
    events=[]
    startup_limit=receipt.get('startup_without_update_limit_minutes')
    resume_steps=receipt.get('resume_steps',0)
    if startup_limit and receipt.get('allocated_utc') and state['max_observed_steps']<=resume_steps and not receipt.get('finished_utc'):
        allocated=datetime.datetime.fromisoformat(receipt['allocated_utc']).timestamp()
        if now-allocated>startup_limit*60:
            events.append(event('preparation_limit',f'分配 GPU 超过 {startup_limit} 分钟仍未确认本次新参数更新；核查缓存、观测链路、传输和启动成本，不因准备计数增长而忽略预算。'))
    if progress.get('steps',0)>resume_steps:
        events.append(event('first_update',f"已确认至少 {progress['steps']} 次参数更新。",global_key=True))
    validations=progress.get('validation',[])
    if validations and validations[-1].get('epoch',len(validations))>receipt.get('validation_epoch_floor',0):
        epoch=validations[-1].get('epoch',len(validations))
        events.append(event('validation',f'第 {epoch} 轮验证结果已产生，请比较定位 AUROC 和分类 AP。',str(epoch),True))
    if not receipt.get('finished_utc'):
        has_progress=any(k in progress for k in ['steps','stage','teacher_videos'])
        if has_progress:
            phase=progress.get('stage','setup')
            marker=(receipt_path,phase,progress.get('steps'),progress.get('epoch_cursor'),
                    progress.get('video_cursor'),progress.get('teacher_videos'))
            if list(marker)!=state.get('marker'):
                state.update(marker=list(marker),last_progress=now)
            minutes={'setup':15,'restoring_assets':65,'teacher_scoring':35,'training':12,'validation':50}.get(phase,20)
            if now-state.get('last_progress',now)>minutes*60:
                events.append(event('stalled',f'{phase} 阶段超过 {minutes} 分钟未见实质进度。',f"{phase}:{progress.get('epoch_cursor')}:{progress.get('steps')}"))
        if online.get('fresh_progress') and has_progress:
            state['last_contact']=now
        elif receipt.get('allocated_utc') and now-state.setdefault('last_contact',now)>15*60:
            events.append(event('monitor_blind','连续 15 分钟无法获得实时状态，需要恢复观测链路。'))
    return events
