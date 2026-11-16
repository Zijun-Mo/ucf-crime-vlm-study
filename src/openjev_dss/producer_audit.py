"""Audit archived numerical producers while allowing explicit checkpoint selection."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def digest(value):
    return hashlib.sha256(value).hexdigest()


def scoring_loop_signature(source):
    tree=ast.parse(source)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For)
           and ast.unparse(n.target)=='(i, record)'
           and ast.unparse(n.iter)=='enumerate(records)']
    assert len(loops)==1,'Missing or ambiguous full-test scoring loop'
    return digest(ast.dump(loops[0],include_attributes=False).encode())


def archived_producer_signature(root,config,receipt):
    root=Path(root)
    archive=root/receipt['source_archive']
    assert digest(archive.read_bytes())==receipt['source_archive_sha256']
    script='research/run_formal_inference.py' if config.get('training_protocol_version')==2 else 'research/run_inference_group.py'
    files={'runtime_code_sha256':'src/openjev_dss/controlled_runtime.py',
           'visual_code_sha256':'src/openjev_dss/study_visuals.py',
           'pair_visual_code_sha256':'src/openjev_dss/study_pair_visuals.py',
           'inference_code_sha256':script}
    with zipfile.ZipFile(archive) as z:
        for key,path in files.items():assert digest(z.read(path))==config[key],path+' differs from its receipt'
        # The model-loading branches intentionally differ. The entire scoring
        # loop, including resume identity checks and raw-logit calculation,
        # must remain identical to the original frozen producer in the archive.
        loop=scoring_loop_signature(z.read(script).decode())
        original=scoring_loop_signature(z.read('research/run_inference_group.py').decode())
        assert loop==original,'Unreviewed change to formal test scoring'
        numeric=digest(z.read('src/openjev_dss/experiment.py'))
        decoder=digest(z.read('src/openjev_dss/runtime.py'))
    return (config['torch'],config['runtime_code_sha256'],config['visual_code_sha256'],
            config['pair_visual_code_sha256'],numeric,decoder,loop)


def audit_formal_training(root,config,receipts,policy):
    from .convergence import record_validation
    root=Path(root)
    candidates=[]
    for path,receipt in receipts:
        if receipt['job']!='train_formal' or receipt['status']!='complete':continue
        result=path.parent/'result.json'
        if result.exists() and digest(result.read_bytes())==config['training_status_sha256']:
            candidates.append((path,receipt,json.loads(result.read_text())))
    assert candidates,'No complete formal training receipt matches this inference checkpoint'
    path,receipt,state=candidates[-1]
    assert state['status']=='complete' and state['test_used'] is False
    assert json.loads((path.parent/'policy.json').read_text())==policy
    assert config['training_policy_sha256']==digest((root/'research/training_protocol_v2.json').read_bytes())
    contract_path=path.parent/'contract.json';contract=json.loads(contract_path.read_text())
    assert digest(contract_path.read_bytes())==config['training_contract_sha256']
    assert contract['policy_sha256']==config['training_policy_sha256']
    assert contract['torch']==config['torch']
    environment=json.loads((path.parent/'environment.json').read_text())
    expected_versions=json.loads((root/'research/gpu_preflight_result.json').read_text())['versions']
    assert environment['versions']==contract['versions']==expected_versions
    preflight=json.loads((path.parent/'preflight.json').read_text())
    assert preflight['status']=='passed' and preflight['test_used'] is False
    assert preflight['vision_gate']['allowed'] and all(v['passed'] for v in preflight['gradients'].values())
    assert preflight['training_execution']==contract['training_execution']
    assert contract['training_execution'] in [{'attention_backend':'auto','gradient_checkpointing':False},
                                               {'attention_backend':'math','gradient_checkpointing':False}]
    assert set(preflight['gradients'])=={'qwen','openjev'}
    for check in preflight['gradients'].values():
        for result in [check,check['dense_repeat'],check['direct_external_gradient']]:
            assert result['relative_l2']<.01 and result['cosine']>.999
        assert check['execution']['original_batch_positions_preserved'] and check['execution']['score_forward_grad_enabled']
    archive=root/receipt['source_archive']
    assert digest(archive.read_bytes())==receipt['source_archive_sha256']
    with zipfile.ZipFile(archive) as z:
        for name,sha in contract['source'].items():assert digest(z.read(name))==sha
    for arm in state['arms'].values():
        replay={}
        for i,row in enumerate(arm['validation'],1):
            assert row['epoch']==i and row['videos']==policy['validation']['videos']
            record_validation(replay,i,row['video_max_ap'],policy['selection'])
        assert replay['stop_reason']=='validation_plateau'
        assert arm['stopped'] and arm['frozen_parameters_unchanged']
        assert arm['best_epoch']==replay['best_epoch'] and arm['best_ap']==replay['best_ap']
        assert arm['stopped_epoch']==replay['stopped_epoch']==len(arm['validation'])
        assert arm['steps']==len(arm['rows'])==len(arm['validation'])*policy['training']['steps_per_epoch']
    arm=state['arms'][config['training_branch']]
    assert arm['selected_artifact']['epoch']==arm['best_epoch']==config['selected_epoch']
    assert arm['selected_artifact']['state_file_sha256']==config['checkpoint_sha256']
    assert state['fingerprint']==config['training_fingerprint']
    return str(path.relative_to(root))
