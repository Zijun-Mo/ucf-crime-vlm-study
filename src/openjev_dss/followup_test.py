"""Offline admission for the single development-selected V4 full test."""
import hashlib,json,math
from pathlib import Path


def frozen_test_gate(root,policy_path):
    root=Path(root);policy_path=Path(policy_path)
    policy=json.loads(policy_path.read_text())
    registration=json.loads(policy_path.with_name('registration.json').read_text())
    selected=json.loads((root/registration.get('selection_source','research/localization_v4/selected_checkpoint.json')).read_text())
    comparison=json.loads((root/registration.get('selection_evidence','results/localization_v4_dev/epoch_2_comparison.json')).read_text())
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest()==registration['policy_sha256'],'Registered policy changed'
    assert selected['qualified'] and comparison['passed'] and comparison['qualified_candidate']
    for k in ['checkpoint_file_sha256','checkpoint_tensor_sha256','selected_epoch']:
        assert policy[k]==registration[k]==selected[k],'Selected checkpoint changed: '+k
    assert policy['selected_epoch']==comparison.get('epoch',selected['selected_epoch'])
    assert registration.get('selection_frozen_before_test',True) and selected.get('selection_frozen',True)
    assert policy['single_preselected_checkpoint'] and not policy['training']
    assert len(policy['records'])==290 and len({r['name'] for r in policy['records']})==290
    assert all(r['split']=='test' for r in policy['records'])
    assert policy['bootstrap']==dict(replicates=1000,seed=42,unit='video')
    assert not (root/'results'/policy['name']/'summary.json').exists(),'Full test already complete; do not run again'
    hashes={};cache_keys=[]
    for row in policy['records']:
        name=row['name'];path=root/policy['baseline_source']/(name+'.json')
        baseline=json.loads(path.read_text());expected=list(range(math.ceil(row['n_frames']/row['fps'])))
        assert baseline['source_sha256']==row['sha256'] and baseline['start_s']==expected
        assert len(baseline['raw_logits'])==len(expected) and all(math.isfinite(z) for z in baseline['raw_logits'])
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        old=json.loads((root/'results/qwen_localization_v2_epoch1_test/raw/predictions'/(name+'.json')).read_text())
        assert old['source_sha256']==row['sha256'] and old['canonical_sha256']==policy['metadata'][name]['canonical_sha256']
        cache_keys.append(old['visual_cache_key'])
    return dict(passed=True,selected_epoch=policy['selected_epoch'],baseline_videos=290,baseline_prediction_sha256=hashes,
        expected_visual_cache_keys=cache_keys,policy_sha256=registration['policy_sha256'],
        selected_checkpoint_file_sha256=policy['checkpoint_file_sha256'],gpu_allocated=False,test_inference_started=False)


def result_marker_gate(policy,result):
    """Validate one worker result; raw-file completeness is checked separately."""
    assert result['status']=='complete' and result['videos']==290 and result['windows']==37188 and result['frames']==1111808
    assert result['fingerprint']==policy['fingerprint'] and result['selected_epoch']==policy['selected_epoch']
    assert result['checkpoint_tensor_sha256']==policy['checkpoint_tensor_sha256'] and result['optimizer_updates']==0
    return True


def completed_test_gate(policy,raw):
    """A result marker alone cannot certify a complete fixed-checkpoint test."""
    raw=Path(raw);result=json.loads((raw/'result.json').read_text());result_marker_gate(policy,result)
    checkpoint=json.loads((raw/'checkpoint_audit.json').read_text())
    assert checkpoint['file_sha256']==policy['checkpoint_file_sha256'] and checkpoint['tensor_sha256']==policy['checkpoint_tensor_sha256']
    assert checkpoint['selected_epoch']==policy['selected_epoch'] and checkpoint['selection_frozen_before_test']
    assert json.loads((raw/'cache_gate.json').read_text())['allowed']
    expected={r['name'] for r in policy['records']};assert len(expected)==290
    assert {p.stem for p in (raw/'predictions').glob('*.json')}==expected
    total=0;hashes={}
    for row in policy['records']:
        path=raw/'predictions'/(row['name']+'.json');p=json.loads(path.read_text())
        assert p['fingerprint']==policy['fingerprint'] and p['source_sha256']==row['sha256']
        assert p['canonical_sha256']==policy['metadata'][row['name']]['canonical_sha256']
        assert p['selected_epoch']==policy['selected_epoch'] and p['checkpoint_tensor_sha256']==policy['checkpoint_tensor_sha256']
        expected_starts=list(range(math.ceil(row['n_frames']/row['fps'])))
        assert p['start_s']==expected_starts and len(p['raw_logits'])==len(expected_starts)
        assert all(math.isfinite(z) for z in p['raw_logits'])
        total+=len(p['raw_logits']);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert total==37188 and sum(r['n_frames'] for r in policy['records'])==1111808
    return dict(passed=True,videos=290,windows=total,frames=1111808,optimizer_updates=0,
        checkpoint_file_sha256=checkpoint['file_sha256'],checkpoint_tensor_sha256=checkpoint['tensor_sha256'],
        result_json_sha256=hashlib.sha256((raw/'result.json').read_bytes()).hexdigest(),prediction_sha256=hashes)
