"""Honor the user's stop; preserve the last committed optimizer and epoch5."""
import os,signal,json,time,hashlib,subprocess,sys
from pathlib import Path
out=Path('/content/ucf_localization_train')
target=out/'operator_user_stop.json'
if target.exists() and json.loads(target.read_text()).get('complete'):
    print('ALREADY_USER_STOPPED',target.read_text())
else:
    handle=json.loads(Path('/content/ucf_background_handle.json').read_text());pid=handle['pid']
    fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    assert pid==3813 and fields[19]=='92595' and fields[0]!='Z','Original worker identity changed; inspect'
    os.kill(pid,signal.SIGSTOP)
    state=json.loads((out/'status.json').read_text())
    assert state['epoch_cursor']==5 and state['steps']>=785 and state['best_epoch']==5
    report=dict(complete=False,started=time.time(),reason='User requested stop after consecutive within-video AUROC declines',worker_pid=pid,committed_steps=state['steps'],completed_epochs=5,selected_epoch=5,converged=False,unvalidated_epoch6_updates=state['steps']-785,resume_file=f"resume_{state['generation']%2}.pt")
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('WORKER_FROZEN',json.dumps(report),flush=True)
    sys.path[:0]=['/content/ucf_deps','/content/ucf_project/src']
    import torch
    from openjev_dss.controlled_training import tensor_digest
    from openjev_dss.cloud_io import checked
    torch.set_num_threads(2)
    resume=out/report['resume_file'];saved=torch.load(resume,map_location='cpu',weights_only=True)
    assert saved['steps']==state['steps'] and saved['fingerprint']==state['fingerprint']
    counters=[int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
    assert counters and all(v==state['steps'] for v in counters)
    selected=out/'epochs/epoch_5/state.pt';actual=tensor_digest(torch.load(selected,map_location='cpu',weights_only=True))
    assert actual=='3f6b974d0622537973d34d76732f5bcfc7d88702cf716037f5db1553fb68eb53'
    assert hashlib.sha256(selected.read_bytes()).hexdigest()=='d205a2e962b584e8b56620f096ab861c0a91988d36c7bb42391ddf5437f467fd'
    report.update(resume_file_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),resume_tensor_sha256=tensor_digest(saved['trainable']),optimizer_steps=counters,selected_tensor_sha256=actual,selected_file_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),resume_rng_preserved=True)
    # Stop an in-flight publisher before writing a terminal pointer ourselves.
    for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
        try:
            cmd=Path(f'/proc/{child}/cmdline').read_bytes()
            if b'/content/ucf-bin/rclone' in cmd and b'ucf_localization_train' in cmd:os.kill(int(child),signal.SIGTERM)
        except (ProcessLookupError,FileNotFoundError):pass
    state.update(status='user_stopped',stage='finished',stop_reason='user_stop_repeated_within_video_decline',converged=False,finished=time.time(),operator_user_stop=report)
    target.write_text(json.dumps(report,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','10s','--timeout','30s','--retries','2']
    remote='gdrive:remaining_20261108_v1/localization_v4_continuation'
    checked(rc+['copyto',str(resume),remote+'/'+resume.name,'--checksum'],timeout=120)
    checked(rc+['copyto',str(resume),remote+f"/user_stop/resume_step_{state['steps']}.pt",'--checksum'],timeout=120)
    metadata=subprocess.run(rc+['lsjson',remote+'/'+resume.name,'--stat','--hash','--log-level','ERROR'],capture_output=True,text=True,timeout=60,check=True)
    observed=json.loads(metadata.stdout)
    assert observed['Size']==resume.stat().st_size and observed['Hashes']['sha256']==report['resume_file_sha256']
    model_metadata=subprocess.run(rc+['lsjson',remote+'/epochs/epoch_5/state.pt','--stat','--hash','--log-level','ERROR'],capture_output=True,text=True,timeout=60,check=True)
    assert json.loads(model_metadata.stdout)['Hashes']['sha256']==report['selected_file_sha256']
    report.update(complete=True,completed=time.time(),resume_remote_verified=True,selected_checkpoint_remote_verified=True)
    target.write_text(json.dumps(report,indent=2)+'\n');state['operator_user_stop']=report
    (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    for name,dest in [('operator_user_stop.json','operator_user_stop.json'),('status.json','user_stop/status.json'),('status.json','status.json')]:
        checked(rc+['copyto',str(out/name),remote+'/'+dest,'--ignore-times'],timeout=60)
    os.kill(pid,signal.SIGKILL)
    print('USER_STOP_DURABLE',json.dumps(report),flush=True)
