"""One Qwen localization run: stage immutable assets, profile, prepare and train."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import torch
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.formal_features import FormalVisualProvider
from openjev_dss.formal_execution import apply_execution
from openjev_dss.learning import atomic_json
from openjev_dss.localization_backward import stream_scores,backward_video
from openjev_dss.localization_objective import selection,objective
from openjev_dss.localization_training import run_localization,gradient_preflight

root=Path('/content/ucf_drive');out=Path('/content/ucf_localization_train');out.mkdir(exist_ok=True)
policy_path=work/os.environ.get('UCF_LOCALIZATION_POLICY','research/localization_v1/policy.json')
policy=json.loads(policy_path.read_text())
from openjev_dss.continuation_control import load_continuation
continuation_path=os.environ.get('UCF_LOCALIZATION_CONTINUATION')
continuation=load_continuation(work,continuation_path,policy_path.relative_to(work).as_posix())
soft=policy['loss'].get('kind')=='pointwise_minmax_soft_v4'
focal=policy['loss'].get('kind')=='pointwise_focal_v3'
pointwise=policy['loss'].get('kind') in ['pointwise_teacher_v2','pointwise_focal_v3','pointwise_minmax_soft_v4']
protocol=json.loads((work/'research/protocol.json').read_text())
deadline=float(os.environ['UCF_DEADLINE']);phase=os.environ.get('UCF_LOCALIZATION_PHASE','prepare')
remote=policy['resources'].get('output_remote','gdrive:remaining_20261108_v1/localization_v1')
if continuation:remote=continuation['output_remote']
cache_remote='gdrive:remaining_20261108_v1/formal_v2/visual_features'
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','3','--retries-sleep','2s']
provider=None;publisher=None


def guard():
    if time.time()>deadline-360:raise TimeoutError('Localization allocation reserve')


def publish():
    global publisher
    if publisher is None:
        from openjev_dss.durable_publish import IncrementalPublisher,remote_inventory
        checked(rc+['mkdir',remote],timeout=120)
        md5=remote_inventory(rc+['lsjson',remote,'--recursive','--files-only','--hash','--log-level','ERROR'])
        def send(names,timeout):
            listing=Path('/content/localization_changed_files.txt');listing.write_text('\n'.join(names)+'\n')
            checked(rc+['copy',str(out),remote,'--files-from',str(listing),'--no-traverse',
                '--checksum','--transfers','4'],timeout=timeout)
        def commit():
            checked(rc+['copyto',str(out/'status.json'),remote+'/status.json','--ignore-times'],timeout=120)
        publisher=IncrementalPublisher(out,send,commit,md5,chunk_size=16)
    publisher()


try:
    # A read-only listing may be absent for a new protocol; rclone's copy is
    # attempted only when the task-scoped folder actually exists.
    if continuation:
        # Use the native API rather than a possibly stale FUSE directory entry.
        from openjev_dss.durable_publish import remote_inventory
        existing=remote_inventory(rc+['lsjson',remote,'--files-only','--max-depth','1','--hash'],timeout=120)
        has_status='status.json' in existing
        restore_remote=remote if has_status else continuation['resume_remote']
        # Bundles are a separate75GB transfer input, not checkpoint history.
        # Restore them once, after the speed gate, through the pinned manifest.
        excluded=['--exclude','/cache_bundle_v1/**'] if continuation.get('visual_cache_bundle') else []
        checked(rc+['copy',restore_remote,str(out),'--transfers','12','--checkers','12']+excluded,timeout=1200)
        assert (out/'status.json').exists(),'Continuation status missing; never start afresh'
        if not has_status:
            restored=json.loads((out/'status.json').read_text())
            assert restored['steps']==continuation['resume_steps']
            checkpoint=out/f"resume_{restored['generation']%2}.pt"
            assert hashlib.md5(checkpoint.read_bytes()).hexdigest()==continuation['resume_file_md5']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==continuation['resume_file_sha256']
        # Requalify the resumed execution sources; retain the previous proof.
        profile=out/'throughput_profile.json'
        if profile.exists():
            row=json.loads(profile.read_text())
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (work/'src/openjev_dss').glob('localization_*.py')}
            if row['source']!=sources:
                profile.rename(out/'throughput_profile_before_continuation.json')
        atomic_json(out/'continuation_control.json',continuation)
    elif (root/remote.removeprefix('gdrive:')).exists():
        checked(rc+['copy',remote,str(out),'--transfers','12','--checkers','12'],timeout=1200)
    if (out/'status.json').exists():
        restored=json.loads((out/'status.json').read_text())
        if restored.get('error'):restored['previous_failure']=restored.pop('error')
        restored.update(status='running',stage='restoring_assets')
        atomic_json(out/'status.json',restored)
    if (out/'policy.json').exists():assert json.loads((out/'policy.json').read_text())==policy
    atomic_json(out/'policy.json',policy)
    manifest_path=root/'weak_dss_learning_10cu_v1/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==policy['train_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    annotations_path=work/policy['validation']['annotations']
    annotations=json.loads(annotations_path.read_text()) if annotations_path.exists() else None
    if phase=='train' and not annotations:raise ValueError('Real development annotation artifact not staged')
    data=Path('/content/ucf_formal_data');(data/'metadata').mkdir(parents=True,exist_ok=True);(data/'canonical').mkdir(exist_ok=True)
    metadata_file=work/'research/formal_training_metadata.json'
    assert hashlib.sha256(metadata_file.read_bytes()).hexdigest()==policy['training_metadata_bundle_sha256']
    metadata=json.loads(metadata_file.read_text())
    for r in records:
        row=metadata['records'][r['name']];assert row['sha256']==r['sha256']
        atomic_json(data/'metadata'/(r['name']+'.json'),row)
    started=time.time();asset=json.loads((work/'research/formal_input_archive.json').read_text())
    assert asset['status']=='passed' and asset['metadata_bundle_sha256']==policy['training_metadata_bundle_sha256']
    archive=data/'canonical_train_val.zip'
    checked(rc+['copyto',asset['remote'],str(archive)],timeout=1800)
    with archive.open('rb') as inp:assert hashlib.file_digest(inp,'sha256').hexdigest()==asset['zip_sha256']
    with zipfile.ZipFile(archive) as pack:
        assert set(pack.namelist())=={r['name'] for r in records}
        assert sum(i.file_size for i in pack.infolist())+32*1024**3 < shutil.disk_usage(data).free
        pack.extractall(data/'canonical')
    archive.unlink()
    staging_seconds=time.time()-started
    os.environ['HF_HUB_DISABLE_XET']='1';torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    from huggingface_hub import snapshot_download
    base=snapshot_download('Qwen/Qwen3.5-2B',revision=policy['backbone_revision'],token=False,
        allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'],max_workers=4)
    runtime=ControlledRuntime(base,protocol['prompts']['P1'],train=True)
    apply_execution({'qwen':runtime},policy['execution'])
    config=runtime.model.peft_config['default'];assert (config.r,config.lora_alpha,config.lora_dropout)==(8,16,0)
    local=Path('/content/ucf_formal_visual');local.mkdir(exist_ok=True)
    guard()
    if not pointwise and (root/'remaining_20261108_v1/formal_v2/visual_features').exists():
        checked(rc+['copy',cache_remote,str(local),'--transfers','4','--checkers','4'],timeout=2400)
    provider=FormalVisualProvider(runtime,data,records,local,policy['backbone_revision'],guard)
    expected=sum((provider.metadata[r['name']]['window_count']+1)*49*2048*2 for r in records)
    cached=sum(p.stat().st_size for p in local.rglob('*.pt'))
    assert expected+12*1024**3 < shutil.disk_usage(local).free+cached,'Full-window cache does not fit the allocation disk'
    def cache_flush():
        size=sum((provider.local/name).stat().st_size for name in provider.pending)
        if size>=1024**3:
            result=provider.flush(lambda cmd,timeout:checked(rc+cmd,timeout=timeout),cache_remote,deadline,reserve=360)
            atomic_json(out/'feature_sync.json',result)
    provider.on_shard=cache_flush
    if pointwise:
        if soft:
            from openjev_dss.localization_soft_profile import profile_soft as profile_pointwise
        elif focal:
            from openjev_dss.localization_focal_profile import profile_focal as profile_pointwise
        else:
            from openjev_dss.localization_pointwise_profile import profile_pointwise
        profile=profile_pointwise(runtime,provider,records,policy,out,staging_seconds,float(os.environ['UCF_QUOTED_RATE_CU_HR']))
        publish();print('LOCALIZATION_POINTWISE_QUALIFIED',json.dumps(profile),flush=True)
        # Large cache transfer follows the speed gate, so a rejected trial does
        # not spend GPU time copying the full cache.
        if continuation and continuation.get('visual_cache_bundle'):
            from openjev_dss.cache_bundle import restore_remote_bundle
            restore_remote_bundle(continuation['visual_cache_bundle'],local,
                lambda cmd,timeout:checked(rc+cmd,timeout=timeout),guard,
                lambda row:atomic_json(out/'cache_bundle_restore.json',row))
        elif (root/'remaining_20261108_v1/formal_v2/visual_features').exists():
            checked(rc+['copy',cache_remote,str(local),'--transfers','8','--checkers','8','--size-only'],timeout=3000)
        if not all((out/'teacher'/(r['name']+'.json')).exists() for r in records):
            legacy=Path('/content/ucf_teacher_v1_import');legacy.mkdir(exist_ok=True)
            source_config_path=work/'research/localization_v4/teacher_cache_source.json'
            source_config=json.loads(source_config_path.read_text()) if soft else dict(remote=policy['resources']['import_teacher_remote'],policy='research/localization_v1/policy.json')
            checked(rc+['copy',source_config['remote'],str(legacy),'--transfers','8'],timeout=300)
            from openjev_dss.localization_teacher_import import import_teacher
            imported=import_teacher(legacy,out/'teacher',manifest,provider.metadata,
                json.loads((work/source_config['policy']).read_text()),policy)
            imported.update(source_remote=source_config['remote'],source_policy=source_config['policy'])
            if soft:
                for name in imported['names']:
                    path=out/'teacher'/(name+'.json');entry=json.loads(path.read_text())
                    entry['imported_from']['protocol']='localization_v2';atomic_json(path,entry)
            if imported['imported_videos'] or not (out/'teacher_import.json').exists():
                atomic_json(out/'teacher_import.json',imported)
            publish();print('LOCALIZATION_TEACHER_IMPORTED',json.dumps(imported),flush=True)
        else:print('LOCALIZATION_TEACHER_CACHE_COMPLETE',len(records),flush=True)
        if soft:
            from openjev_dss.teacher_cache_gate import require_complete_teacher
            from openjev_dss.localization_teacher_import import fingerprint
            gate=require_complete_teacher(records,out/'teacher',provider.metadata,fingerprint(manifest,policy))
            atomic_json(out/'native_teacher_cache_gate.json',gate);publish()
    if not pointwise and not (out/'throughput_profile.json').exists():
        fixture=min((r for r in records if r['split']=='train' and r['label']),key=lambda r:r['size'])
        view=provider(fixture,0,True);features=view[:min(4,len(view))]
        teacher=[-3.]*(len(features)//2)+[4.]*(len(features)-len(features)//2)
        chosen=selection(teacher,dict(threshold=0.,scale=1.),policy['selection'])
        gradients=gradient_preflight(runtime,features,chosen,out/'gradient_preflight.json')
        timings=[]
        for label in [0,1]:
            candidate=min((r for r in records if r['split']=='train' and r['label']==label and 128<=provider.metadata[r['name']]['window_count']<=512),
                          key=lambda r:(provider.metadata[r['name']]['window_count'],r['name']))
            view=provider(candidate,0,True);short=view[:min(64,len(view))]
            runtime.model.eval();runtime.head.eval();torch.cuda.synchronize();begin=time.time()
            with runtime.model.disable_adapter():t=stream_scores(runtime.forward,short).tolist()
            torch.cuda.synchronize();forward_s=time.time()-begin
            selected=selection(t,dict(threshold=float(torch.tensor(t).median()),scale=1.),policy['selection'])
            runtime.model.train();runtime.head.train();runtime.model.zero_grad(set_to_none=True)
            begin=time.time();value,info=backward_video(runtime.forward,short,lambda z:objective(z,label,selected))
            torch.cuda.synchronize();training_s=time.time()-begin;runtime.model.zero_grad(set_to_none=True)
            timings.append(dict(name=candidate['name'],label=label,windows=len(short),
                teacher_seconds=forward_s,student_seconds=training_s,execution=info))
        total=sum(provider.metadata[r['name']]['window_count'] for r in records)
        by_class={y:sum(provider.metadata[r['name']]['window_count'] for r in records if r['split']=='train' and r['label']==y) for y in [0,1]}
        forward_per_window=sum(r['teacher_seconds'] for r in timings)/sum(r['windows'] for r in timings)
        epoch_estimate=sum(by_class[r['label']]*r['student_seconds']/r['windows'] for r in timings)
        val_windows=sum(provider.metadata[r['name']]['window_count'] for r in records if r['split']=='val')
        rate=float(os.environ['UCF_QUOTED_RATE_CU_HR'])
        profile=dict(status='passed',gpu=torch.cuda.get_device_name(),measurements=timings,gradient_gate=gradients['passed'],
            teacher_forward_only_hours=total*forward_per_window/3600,
            epoch_training_only_hours=epoch_estimate/3600,validation_hours=val_windows*forward_per_window/3600,
            two_epoch_minimum_estimated_cu=(total*forward_per_window+2*(epoch_estimate+val_windows*forward_per_window))*rate/3600,
            rate_cu_hr=rate,canonical_staging_s=staging_seconds,feature_cache_bytes_upper_bound=expected,
            limitation='Small training-fixture timing extrapolation; excludes I/O, setup, checkpoint publication, and selection-dependent backward variation',test_used=False)
        atomic_json(out/'throughput_profile.json',profile);publish();print('LOCALIZATION_PROFILE',json.dumps(profile),flush=True)
        if profile['two_epoch_minimum_estimated_cu']>policy['resources']['training_cu_budget']:
            raise RuntimeError('Measured two-epoch forecast exceeds fixed localization credit budget')
    contract=dict(policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        vision_gate=provider.gate,backbone_revision=policy['backbone_revision'],template_sha256=runtime.template_sha256,
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (work/'src/openjev_dss').glob('localization_*.py')},
        test_used=False,phase=phase)
    atomic_json(out/'contract.json',contract);publish()
    def load_reviewed_annotations():
        path=root/('remaining_20261108_v1/localization_dev_v2/dev_annotations.json' if (focal or soft) else 'remaining_20261108_v1/localization_dev/dev_annotations.json')
        if not path.exists():return annotations
        observed=json.loads(path.read_text())
        atomic_json(out/'dev_annotations_used.json',observed)
        return observed
    state=run_localization(runtime,manifest,provider,out,deadline,policy,phase,annotations,publish,load_reviewed_annotations,
        continuation=continuation)
    if provider.pending:
        result=provider.flush(lambda cmd,timeout:checked(rc+cmd,timeout=timeout),cache_remote,deadline,reserve=180)
        atomic_json(out/'feature_sync.json',result)
    atomic_json(out/'runtime_receipt.json',dict(status=state['status'],phase=phase,gpu=torch.cuda.get_device_name(),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,test_used=False))
    publish();print('LOCALIZATION_RESULT',json.dumps({k:v for k,v in state.items() if k!='rows'}),flush=True)
except Exception:
    import traceback
    error=sanitize(traceback.format_exc());(out/'worker_error.txt').write_text(error)
    try:publish()
    except Exception:pass
    raise
