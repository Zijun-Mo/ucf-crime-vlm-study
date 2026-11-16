"""Score one preselected localization checkpoint; evaluation happens on CPU."""
import hashlib,json,math,os,sys,time,traceback
from pathlib import Path
work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import numpy as np
import torch
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.learning import atomic_json
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.controlled_training import tensor_digest
from openjev_dss.storyboard_cached import cache_signature,PairFeatureView
from openjev_dss.study_pair_visuals import GatedPairVisualStore
from openjev_dss.experiment import score_windows

out=Path('/content/ucf_followup');out.mkdir(exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
policy=json.loads((work/os.environ.get('UCF_FOLLOWUP_POLICY','research/localization_v2_test/policy.json')).read_text())
deadline=float(os.environ['UCF_DEADLINE']);remote=policy['output_remote'];started=time.time()
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','2','--retries-sleep','2s']
state=dict(status='running',stage='initializing',completed_videos=0,total_videos=290,scored_windows=0,
    optimizer_updates=0,started=started,test_used=True,fingerprint=policy['fingerprint'])
def status(stage,**extra):
    state.update(stage=stage,updated=time.time(),**extra);atomic_json(out/'status.json',state)
def guard(reserve=180):
    if time.time()>deadline-reserve:raise TimeoutError('Checkpoint test durable-save reserve')
def publish():
    checked(rc+['copy',str(out),remote,'--transfers','8','--checkers','8','--checksum'],timeout=150)
def copy(source,target,timeout=240):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    checked(rc+['copyto',source,str(target)],timeout=timeout);return target
try:
    assert torch.cuda.get_device_name()=='NVIDIA L4'
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    checked(rc+['mkdir',remote],timeout=60)
    checked(rc+['copy',remote,str(out),'--transfers','8'],timeout=180)
    if (out/'policy.json').exists():assert json.loads((out/'policy.json').read_text())==policy
    atomic_json(out/'policy.json',policy)
    checkpoint=Path('/content/localization_selected')
    checked(rc+['copy',policy['checkpoint_remote'],str(checkpoint),'--transfers','8'],timeout=240)
    state_path=checkpoint/'state.pt'
    assert hashlib.sha256(state_path.read_bytes()).hexdigest()==policy['checkpoint_file_sha256']
    snapshot=torch.load(state_path,map_location='cpu',weights_only=True)
    assert tensor_digest(snapshot)==policy['checkpoint_tensor_sha256']
    atomic_json(out/'checkpoint_audit.json',dict(file_sha256=policy['checkpoint_file_sha256'],
        tensor_sha256=tensor_digest(snapshot),selected_epoch=policy['selected_epoch'],selection_frozen_before_test=True))
    os.environ['HF_HUB_DISABLE_XET']='1';status('loading_model')
    from huggingface_hub import snapshot_download
    base=snapshot_download(policy['model_id'],revision=policy['revision'],token=False,
        allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'],max_workers=4)
    runtime=ControlledRuntime(base,policy['prompt'],4,adapter=str(checkpoint/'adapter'),state=state_path)
    runtime.model.eval();runtime.head.eval();assert runtime.trainable_parameters==0
    # Re-verify exact pair reconstruction on two TRAINING fixtures with the
    # selected adapter. Test labels and test logits cannot admit a cache route.
    fixtures=policy['cache_gate_fixtures'];metadata=policy['cache_gate_metadata']
    canon=Path('/content/localization_gate');canon.mkdir(exist_ok=True)
    for row in fixtures:
        f=copy(policy['cache_gate_canonical_remote']+'/'+row['name'],canon/row['name'])
        assert hashlib.sha256(f.read_bytes()).hexdigest()==metadata[row['name']]['canonical_sha256']
    store=GatedPairVisualStore(runtime,canon,metadata,'/content/localization_gate_features',policy['revision'],guard=guard)
    gate=store.gate(fixtures,4);assert gate['allowed'],'Pair cache admission failed; no silent numerical fallback'
    atomic_json(out/'cache_gate.json',gate)
    status('scoring');publish()
    for number,row in enumerate(policy['records']):
        guard();name=row['name'];target=out/'predictions'/f'{name}.json';meta=policy['metadata'][name]
        if target.exists():
            p=json.loads(target.read_text());assert p['fingerprint']==policy['fingerprint'] and p['source_sha256']==row['sha256']
        else:
            signature,key=cache_signature(runtime,row,meta,policy['revision'])
            status('staging_visual_cache',active_video=name)
            path=copy(policy['visual_cache_remote']+'/'+key,Path('/content/test_features')/key)
            view=PairFeatureView(path,signature,meta['window_count'])
            logits=[];status('scoring',active_video=name,active_window_cursor=0)
            with torch.inference_mode():
                for begin in range(0,len(view),4):
                    guard();features=view[begin:begin+4]
                    logits.extend(score_windows(runtime.forward,features,4).tolist())
                    if begin%128==0:status('scoring',active_video=name,active_window_cursor=begin)
            assert len(logits)==meta['window_count'] and np.isfinite(logits).all()
            p=dict(name=name,fingerprint=policy['fingerprint'],source_sha256=row['sha256'],
                canonical_sha256=meta['canonical_sha256'],start_s=list(range(len(logits))),raw_logits=logits,
                selected_epoch=policy['selected_epoch'],checkpoint_tensor_sha256=policy['checkpoint_tensor_sha256'],visual_cache_key=key)
            atomic_json(target,p);del features,view;path.unlink()
        assert len(p['raw_logits'])==math.ceil(row['n_frames']/row['fps'])
        state.update(completed_videos=number+1,scored_windows=state['scored_windows']+len(p['raw_logits']))
        status('scoring',active_video=name)
        if (number+1)%5==0:publish()
        print('LOCALIZATION_TEST_VIDEO',number+1,name,flush=True)
    assert state['completed_videos']==290 and state['scored_windows']==37188
    atomic_json(out/'result.json',dict(status='complete',videos=290,windows=37188,frames=1111808,
        checkpoint_tensor_sha256=policy['checkpoint_tensor_sha256'],selected_epoch=policy['selected_epoch'],
        fingerprint=policy['fingerprint'],optimizer_updates=0,worker_elapsed_s=time.time()-started,
        evaluation_pending='CPU paired bootstrap after GPU release'))
    status('finished',status='complete');publish();print('LOCALIZATION_TEST_COMPLETE',flush=True)
except Exception:
    status('failed',status='error',error=sanitize(traceback.format_exc()))
    try:publish()
    except Exception:pass
    raise
