import hashlib,json
from pathlib import Path
import pytest
from openjev_dss.followup_test import frozen_test_gate


def fixture(root):
    def write(name,data):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));return p
    records=[];metadata={}
    for i in range(290):
        name=f'v{i}';records.append(dict(name=name,sha256=name,split='test',n_frames=4,fps=1))
        metadata[name]=dict(canonical_sha256='canonical-'+name)
        write('results/baseline/'+name+'.json',dict(source_sha256=name,start_s=[0,1,2,3],raw_logits=[-.5,0.,1.,2.]))
        write('results/qwen_localization_v2_epoch1_test/raw/predictions/'+name+'.json',dict(source_sha256=name,canonical_sha256='canonical-'+name,visual_cache_key=name+'.pt'))
    selected=dict(qualified=True,selected_epoch=2,checkpoint_file_sha256='file',checkpoint_tensor_sha256='tensor')
    write('research/localization_v4/selected_checkpoint.json',selected)
    write('results/localization_v4_dev/epoch_2_comparison.json',dict(passed=True,qualified_candidate=True))
    policy={**selected,'single_preselected_checkpoint':True,'training':False,'records':records,'metadata':metadata,
        'bootstrap':dict(replicates=1000,seed=42,unit='video'),'name':'candidate','baseline_source':'results/baseline'}
    path=write('research/localization_v4_test/policy.json',policy)
    write('research/localization_v4_test/registration.json',{**selected,'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def test_complete_baseline_and_pinned_checkpoint_are_admitted_without_gpu(tmp_path):
    p=fixture(tmp_path);gate=frozen_test_gate(tmp_path,p)
    assert gate['passed'] and len(gate['expected_visual_cache_keys'])==290
    assert not gate['gpu_allocated'] and not gate['test_inference_started']


def test_substituting_checkpoint_after_registration_is_rejected(tmp_path):
    p=fixture(tmp_path);selected=tmp_path/'research/localization_v4/selected_checkpoint.json'
    value=json.loads(selected.read_text());value['checkpoint_tensor_sha256']='substituted';selected.write_text(json.dumps(value))
    with pytest.raises(AssertionError,match='checkpoint changed'):frozen_test_gate(tmp_path,p)


def test_incomplete_or_wrong_source_baseline_is_rejected(tmp_path):
    p=fixture(tmp_path);base=tmp_path/'results/baseline/v0.json';value=json.loads(base.read_text())
    value['source_sha256']='different-video';base.write_text(json.dumps(value))
    with pytest.raises(AssertionError):frozen_test_gate(tmp_path,p)


def test_completed_marker_without_all_predictions_is_rejected(tmp_path):
    from openjev_dss.followup_test import completed_test_gate
    p=fixture(tmp_path);policy=json.loads(p.read_text());policy['fingerprint']='fp'
    raw=tmp_path/'raw';raw.mkdir();(raw/'predictions').mkdir()
    (raw/'result.json').write_text(json.dumps(dict(status='complete',videos=290,windows=37188,frames=1111808,fingerprint='fp',selected_epoch=2,checkpoint_tensor_sha256='tensor',optimizer_updates=0)))
    (raw/'checkpoint_audit.json').write_text(json.dumps(dict(file_sha256='file',tensor_sha256='tensor',selected_epoch=2,selection_frozen_before_test=True)))
    (raw/'cache_gate.json').write_text('{"allowed":true}')
    with pytest.raises(AssertionError):completed_test_gate(policy,raw)


def test_wrong_marker_cannot_be_used_as_transport_recovery():
    from openjev_dss.followup_test import result_marker_gate
    policy=dict(fingerprint='fp',selected_epoch=2,checkpoint_tensor_sha256='tensor')
    result=dict(status='complete',videos=290,windows=37188,frames=1111808,fingerprint='fp',selected_epoch=2,checkpoint_tensor_sha256='tensor',optimizer_updates=0)
    assert result_marker_gate(policy,result)
    for change in [dict(videos=289),dict(optimizer_updates=1),dict(selected_epoch=1),dict(fingerprint='other'),dict(checkpoint_tensor_sha256='other')]:
        with pytest.raises(AssertionError):result_marker_gate(policy,{**result,**change})


@pytest.mark.parametrize('epoch',[5,9])
def test_newly_authorized_checkpoint_uses_its_own_frozen_selection(tmp_path,epoch):
    original=fixture(tmp_path);policy=json.loads(original.read_text())
    selected=dict(qualified=True,selected_epoch=epoch,selection_frozen=True,checkpoint_file_sha256=f'file{epoch}',checkpoint_tensor_sha256=f'tensor{epoch}')
    run='localization_v4_continuation' if epoch==5 else 'localization_v4_plateau'
    source=tmp_path/f'research/{run}/selected_checkpoint.json';source.parent.mkdir(parents=True);source.write_text(json.dumps(selected))
    evidence=tmp_path/f'comparison{epoch}.json';evidence.write_text(json.dumps(dict(epoch=epoch,passed=True,qualified_candidate=True)))
    policy.update(selected_epoch=epoch,checkpoint_file_sha256=f'file{epoch}',checkpoint_tensor_sha256=f'tensor{epoch}',name=f'candidate{epoch}')
    experiment='localization_v4_epoch5_test' if epoch==5 else 'localization_v4_plateau_test'
    folder=tmp_path/f'research/{experiment}';folder.mkdir();path=folder/'policy.json';path.write_text(json.dumps(policy))
    registration={**selected,'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selection_source':str(source.relative_to(tmp_path)),'selection_evidence':evidence.name,'selection_frozen_before_test':True}
    (folder/'registration.json').write_text(json.dumps(registration))
    assert frozen_test_gate(tmp_path,path)['selected_epoch']==epoch
    assert json.loads((tmp_path/'research/localization_v4/selected_checkpoint.json').read_text())['selected_epoch']==2
    selected['selection_frozen']=False;source.write_text(json.dumps(selected))
    with pytest.raises(AssertionError):frozen_test_gate(tmp_path,path)
