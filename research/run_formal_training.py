"""Train both formal MIL arms to a validation plateau; keep test labels out."""
import hashlib,json,os,shutil,sys,time,traceback,zipfile
from pathlib import Path
work=Path('/content/ucf_project');sys.path[:0]=['/content/ucf_deps',str(work/'src')]
import torch
from openjev_dss.cloud_io import checked,sanitize
from openjev_dss.checkpoint_sync import publish_training_bundle
from openjev_dss.controlled_runtime import ControlledRuntime
from openjev_dss.formal_features import FormalVisualProvider
from openjev_dss.formal_execution import select_execution
from openjev_dss.formal_training import run_formal_mil
from openjev_dss.controlled_training import tensor_digest
from openjev_dss.training import trainable_state

root=Path('/content/ucf_drive');policy=json.loads((work/'research/training_protocol_v2.json').read_text())
protocol=json.loads((work/'research/protocol.json').read_text())
deadline=float(os.environ['UCF_DEADLINE']);epoch_limit=int(os.environ.get('UCF_FORMAL_EPOCH_LIMIT','10'))
assert epoch_limit>=policy['epoch_budget']['initial'] and (epoch_limit-policy['epoch_budget']['initial'])%policy['epoch_budget']['extension_block']==0
out=Path('/content/ucf_formal_train');out.mkdir(exist_ok=True)
remote='gdrive:remaining_20261108_v1/formal_v2/training'
cache_remote='gdrive:remaining_20261108_v1/formal_v2/visual_features'
rc=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','--contimeout','15s','--timeout','60s','--retries','3','--retries-sleep','2s']
provider=None

def guard():
    if time.time()>deadline-240:raise TimeoutError('Formal allocation reserve')

def publish():
    publish_training_bundle(out,remote,lambda command:checked(rc+command,timeout=180))
    if provider is not None:
        try:
            result=provider.flush(lambda command,timeout:checked(rc+command,timeout=timeout),cache_remote,deadline)
            (out/'feature_cache_sync.json').write_text(json.dumps(result,indent=2)+'\n')
        except Exception as error:
            # Cache durability affects recomputation cost, not optimizer validity.
            (out/'feature_cache_sync_warning.txt').write_text(sanitize(str(error)))

try:
    existing=root/'remaining_20261108_v1/formal_v2/training'
    if existing.exists():checked(rc+['copy',remote,str(out),'--transfers','4'],timeout=240)
    policy_hash=hashlib.sha256((work/'research/training_protocol_v2.json').read_bytes()).hexdigest()
    if (out/'policy.json').exists():assert json.loads((out/'policy.json').read_text())==policy
    (out/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    manifest_path=root/'weak_dss_learning_10cu_v1/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==policy['train_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());assert manifest['counts']=={'train':1256,'val':163,'cal':163}
    records=[r for r in manifest['records'] if r['split'] in ['train','val']]
    # Materialize the small canonical videos with parallel native transfers,
    # before loading the models; the provider still checks every source hash.
    staging_started=time.time()
    data=Path('/content/ucf_formal_data');data.mkdir(exist_ok=True)
    (data/'metadata').mkdir(exist_ok=True);(data/'canonical').mkdir(exist_ok=True)
    metadata_file=work/'research/formal_training_metadata.json'
    assert hashlib.sha256(metadata_file.read_bytes()).hexdigest()==policy['training_metadata_bundle_sha256']
    metadata_bundle=json.loads(metadata_file.read_text())
    assert metadata_bundle['train_manifest_sha256']==policy['train_manifest_sha256']
    assert set(metadata_bundle['records'])=={r['name'] for r in records}
    for r in records:
        meta=metadata_bundle['records'][r['name']];assert meta['sha256']==r['sha256']
        (data/'metadata'/(r['name']+'.json')).write_text(json.dumps(meta))
    archive_receipt=work/'research/formal_input_archive.json'
    staging_mode='native_files'
    if archive_receipt.exists():
        asset=json.loads(archive_receipt.read_text());assert asset['status']=='passed'
        assert asset['metadata_bundle_sha256']==hashlib.sha256(metadata_file.read_bytes()).hexdigest()
        archive=data/'canonical_train_val.zip'
        checked(rc+['copyto',asset['remote'],str(archive)],timeout=min(1800,max(180,deadline-time.time()-600)))
        with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==asset['zip_sha256']
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist())=={r['name'] for r in records}
            canonical_bytes=sum(i.file_size for i in z.infolist())
            assert canonical_bytes+32*1024**3<shutil.disk_usage(data).free
            z.extractall(data/'canonical')
        staging_mode='verified_stored_zip'
    else:
        listing=data/'canonical_files.txt';listing.write_text('\n'.join(r['name'] for r in records)+'\n')
        canonical_inventory=json.loads(checked(rc+['size','gdrive:weak_dss_smoke_v1/canonical','--files-from',str(listing),'--json','--log-level','ERROR'],timeout=90).stdout)
        assert canonical_inventory['count']==len(records),'Missing canonical training/validation videos'
        canonical_bytes=canonical_inventory['bytes']
        assert canonical_bytes+32*1024**3<shutil.disk_usage(data).free,'Canonical staging exceeds disk readiness budget'
        checked(rc+['copy','gdrive:weak_dss_smoke_v1/canonical',str(data/'canonical'),'--files-from',str(listing),'--transfers','8','--checkers','8'],timeout=min(1800,max(180,deadline-time.time()-600)))
    assert all((data/'canonical'/r['name']).is_file() for r in records)
    assert sum((data/'canonical'/r['name']).stat().st_size for r in records)==canonical_bytes
    staging={'source':'gdrive:weak_dss_smoke_v1/canonical','videos':len(records),'bytes':canonical_bytes,
             'seconds':time.time()-staging_started,'mode':staging_mode,'native_transfers':8,'hash_verification':'provider checks every file before feature extraction'}
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import snapshot_download
    base=snapshot_download('Qwen/Qwen3.5-2B',revision='15852e8c16360a2fea060d615a32b45270f8a8fc',token=False,
                          allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','*.model'],max_workers=4)
    runtimes={'qwen':ControlledRuntime(base,protocol['prompts']['P1'],train=True),
              'openjev':ControlledRuntime(base,protocol['prompts']['P1'],train=True,
                  published_checkpoint=root/'models/openjev/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint')}
    for rt in runtimes.values():
        config=rt.model.peft_config['default'];assert (config.r,config.lora_alpha,config.lora_dropout)==(8,16,0)
    local=Path('/content/ucf_formal_visual');local.mkdir(exist_ok=True)
    cloud_cache=root/'remaining_20261108_v1/formal_v2/visual_features'
    if cloud_cache.exists():
        checked(rc+['copy',cache_remote,str(local),'--transfers','4','--checkers','4','--min-size','1'],timeout=min(1800,max(180,deadline-time.time()-600)))
    provider=FormalVisualProvider(runtimes['qwen'],data,records,local,'15852e8c16360a2fea060d615a32b45270f8a8fc',guard)
    fixtures=[min((r for r in records if r['split']=='train' and r['label']==y),key=lambda r:(r['size'],r['name'])) for y in [0,1]]
    bags=[]
    for r in fixtures:
        f=provider(r,0,False);bags.append(f[:2] if len(f)>=2 else f.expand(2,*f.shape[1:]))
    training_execution,gradients=select_execution(runtimes,bags,out)
    example_bytes=bags[0].element_size()
    expected=sum((provider.metadata[r['name']]['window_count']+1)*49*2048*example_bytes for r in records)
    assert expected+32*1024**3<shutil.disk_usage(local).free+sum(p.stat().st_size for p in local.rglob('*.pt')),'Frozen feature cache exceeds disk readiness budget'
    environment=json.loads(Path('/content/ucf_environment_ready.json').read_text())
    assert environment['ready'] and environment['versions']['torch']==str(torch.__version__)
    contract={'policy_sha256':policy_hash,'torch':str(torch.__version__),'versions':environment['versions'],'gpu':torch.cuda.get_device_name(),
              'backbone_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
              'decision_adapter_revision':'0c7aa498b1627be8da4acf34c863ff0ee0a92785',
              'visual_signature':provider.store.signature,
              'training_execution':training_execution,
              'metadata_bundle_sha256':hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
              'initial_trainables':{n:tensor_digest(trainable_state(rt.model,rt.head)) for n,rt in runtimes.items()},
              'template_sha256':{n:rt.template_sha256 for n,rt in runtimes.items()},
              'source':{p:hashlib.sha256((work/p).read_bytes()).hexdigest() for p in [
                  'research/run_formal_training.py','src/openjev_dss/formal_training.py','src/openjev_dss/formal_features.py','src/openjev_dss/convergence.py',
                  'src/openjev_dss/controlled_runtime.py','src/openjev_dss/study_visuals.py','src/openjev_dss/study_pair_visuals.py',
                  'src/openjev_dss/training.py','src/openjev_dss/losses.py','src/openjev_dss/data.py',
                  'src/openjev_dss/learning.py','src/openjev_dss/experiment.py','src/openjev_dss/formal_preflight.py','src/openjev_dss/formal_backward.py','src/openjev_dss/formal_execution.py',
                  'src/openjev_dss/runtime.py']}}
    contract_path=out/'contract.json'
    if contract_path.exists():assert json.loads(contract_path.read_text())==contract,'Formal numerical contract changed'
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    (out/'preflight.json').write_text(json.dumps({'status':'passed','vision_gate':provider.gate,'gradients':gradients,'training_execution':training_execution,
        'feature_bytes_estimate_full_upper_bound':expected,'cache_format':'immutable_aligned_pair_shards_v1',
        'validation_window_batch':4,'max_open_shards':2,'shard_max_pairs':1024,'canonical_staging':staging,'trainable_parameters':{n:rt.trainable_parameters for n,rt in runtimes.items()},
        'test_used':False},indent=2)+'\n')
    publish()
    def flush_large_cache():
        if sum((provider.local/name).stat().st_size for name in provider.pending)<1024**3:return
        try:
            result=provider.flush(lambda command,timeout:checked(rc+command,timeout=timeout),cache_remote,deadline)
            (out/'feature_cache_sync.json').write_text(json.dumps(result,indent=2)+'\n')
        except Exception as error:
            (out/'feature_cache_sync_warning.txt').write_text(sanitize(str(error)))
    provider.on_shard=flush_large_cache
    state=run_formal_mil(runtimes,manifest,provider,out,deadline,policy,epoch_limit,publish)
    (out/'runtime_receipt.json').write_text(json.dumps({'status':state['status'],'gpu':torch.cuda.get_device_name(),
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'epoch_limit':epoch_limit,'test_used':False},indent=2)+'\n')
    publish();print(json.dumps(state,indent=2),flush=True)
except Exception:
    error=sanitize(traceback.format_exc())
    (out/'worker_error.txt').write_text(error)
    try:publish()
    except Exception:pass
    raise
