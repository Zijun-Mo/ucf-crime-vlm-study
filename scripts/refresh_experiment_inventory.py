"""Rebuild the CPU-only authoritative experiment inventory from actual artifacts."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.learning import atomic_json
from openjev_dss.followup_budget import spending

def main():
    folder=ROOT/'results/future_work_v1';folder.mkdir(exist_ok=True)
    conditions=[]
    for path in sorted((ROOT/'results/study_formal_v2').glob('*.json')):
        report=json.loads(path.read_text())
        if not ('metrics' in report and 'videos' in report):continue
        assert report['videos']==290 and report['frames']==1111808
        conditions.append(dict(name=path.stem,status='complete',videos=report['videos'],frames=report['frames'],
            metrics={k:v['value'] for k,v in report['metrics'].items()},source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(conditions)==13
    storyboard=json.loads((ROOT/'results/storyboard_context_formal_v1/summary.json').read_text())
    conditions.append(dict(name='qwen_storyboard_context_formal_v1',status=storyboard['status'],
        metrics=storyboard['metrics']['storyboard'],source='results/storyboard_context_formal_v1/summary.json'))
    result=ROOT/'results/qwen_localization_v2_epoch1_test/summary.json'
    if result.exists():
        report=json.loads(result.read_text());conditions.append(dict(name='qwen_localization_v2_epoch1_test',status='complete',
            metrics=report['metrics']['candidate'],source=str(result.relative_to(ROOT))))
    else:
        receipts=sorted((ROOT/'research/localization_v2_test').glob('20*/receipt.json'))
        receipt=json.loads(receipts[-1].read_text()) if receipts else {}
        conditions.append(dict(name='qwen_localization_v2_epoch1_test',status=receipt.get('status','pending'),
            selected_epoch=1,source=str(receipts[-1].relative_to(ROOT)) if receipts else None))
    soft_path=ROOT/'results/qwen_localization_v4_epoch2_test/summary.json'
    soft=json.loads(soft_path.read_text()) if soft_path.exists() else None
    if soft:conditions.append(dict(name='qwen_localization_v4_epoch2_test',status='complete',videos=290,frames=1111808,metrics=soft['metrics']['candidate'],source=str(soft_path.relative_to(ROOT))))
    final_path=ROOT/'results/qwen_localization_v4_epoch5_test/summary.json'
    final=json.loads(final_path.read_text()) if final_path.exists() else None
    final_registration=ROOT/'research/localization_v4_epoch5_test/registration.json'
    final_status=None
    if final:
        final_status='complete';conditions.append(dict(name='qwen_localization_v4_epoch5_test',status='complete',selected_epoch=5,videos=290,frames=1111808,metrics=final['metrics']['candidate'],source=str(final_path.relative_to(ROOT))))
    elif final_registration.exists():
        attempts=sorted(final_registration.parent.glob('20*/receipt.json'))
        actual=json.loads(attempts[-1].read_text()) if attempts else {}
        final_status=actual.get('status','registered_not_started')
        if final_status=='complete':final_status='inference_complete_scoring_pending'
        conditions.append(dict(name='qwen_localization_v4_epoch5_test',status=final_status,selected_epoch=5,expected_videos=290,expected_windows=37188,expected_frames=1111808,source=str(final_registration.relative_to(ROOT))))
    plateau_test_root=ROOT/'research/localization_v4_plateau_test'
    plateau_summary=ROOT/'results/qwen_localization_v4_plateau_test/summary.json'
    plateau_test_status=None
    if plateau_summary.exists():
        report=json.loads(plateau_summary.read_text());plateau_test_status='complete'
        conditions.append(dict(name='qwen_localization_v4_plateau_test',status='complete',selected_epoch=report['selected_epoch'],videos=290,frames=1111808,metrics=report['metrics']['candidate'],source=str(plateau_summary.relative_to(ROOT))))
    elif (plateau_test_root/'registration.json').exists():
        registration=json.loads((plateau_test_root/'registration.json').read_text())
        attempts=sorted(plateau_test_root.glob('20*/receipt.json'))
        actual=json.loads(attempts[-1].read_text()) if attempts else {}
        plateau_test_status=actual.get('status','registered_not_started')
        if plateau_test_status=='complete':plateau_test_status='inference_complete_scoring_pending'
        conditions.append(dict(name='qwen_localization_v4_plateau_test',status=plateau_test_status,selected_epoch=registration['selected_epoch'],expected_videos=290,expected_windows=37188,expected_frames=1111808,source=str((plateau_test_root/'registration.json').relative_to(ROOT))))
    preflight=ROOT/'research/localization_v3/cpu_preflight.json'
    gate=json.loads(preflight.read_text()) if preflight.exists() else {'status':'pending'}
    usage=[]
    for path in (ROOT/'research/localization_dev_v2').glob('*.json'):
        try:d=json.loads(path.read_text())
        except ValueError:continue
        if d.get('usage'):usage.append(dict(source=str(path.relative_to(ROOT)),status=d.get('status'),usage=d['usage']))
    atomic_json(folder/'hydrogen_usage.json',dict(records=usage,monetary_price_unknown=True,colab_cu=0,
        note='Model tokens recorded separately; retries and failed invocations are retained, not hidden'))
    output=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),proposal_required_conditions=13,
        proposal_required_complete=13,conditions=conditions,v3=dict(status=gate['status'],preflight='research/localization_v3/cpu_preflight.json',
        trained=False if gate.get('passed') is not True else 'check actual optimizer receipt; a passed gate is not training'),
        annotation_progress=dict(new_native_drafts=len(list((ROOT/'research/localization_dev_v2/hydrogen_drafts').glob('*.json'))),
            new_refined_drafts=len(list((ROOT/'research/localization_dev_v2/hydrogen_refined').glob('*.json'))),
            reviewed_artifact_exists=(ROOT/'research/localization_dev_v2/dev_annotations.json').exists()),budget=spending(ROOT),
        boundaries=dict(poster_production=False,proposal_unchanged=True,multi_seed=False,ablations=False,gpu_allowlist=['L4']))
    if soft:output['v4']=dict(status='complete',trained=True,selected_epoch=2,completed_validation_epochs=2,full_test_complete=True,summary=str(soft_path.relative_to(ROOT)),training_gpu_released=True,test_gpu_released=soft['gpu_released'],converged=False)
    continuation=ROOT/'research/localization_v4_continuation/control.json'
    if continuation.exists():
        launch=continuation.with_name('launch.json');live=json.loads(launch.read_text()) if launch.exists() else {}
        output['v4_continuation']=dict(authorization=str(continuation.relative_to(ROOT)),status=live.get('status','authorized'),
            launch=str(launch.relative_to(ROOT)),additional_training_limit_cu=20.,full_test_automatically_authorized=False,
            actual_new_update_verified=live.get('actual_new_update_verified',False))
        if final_registration.exists():output['v4_continuation'].update(status='user_stopped_epoch5_test_'+str(final_status),training_stopped_by_user=True,converged=False,selected_epoch=5,explicit_epoch5_test_authorized=True,full_test_complete=final is not None,full_test_registration=str(final_registration.relative_to(ROOT)))
    plateau=ROOT/'research/localization_v4_plateau/control.json'
    if plateau.exists():
        control=json.loads(plateau.read_text());launch=plateau.with_name('launch.json')
        live=json.loads(launch.read_text()) if launch.exists() else {}
        output['v4_plateau']=dict(authorization=str(plateau.relative_to(ROOT)),status=live.get('status','authorized'),
            launch=str(launch.relative_to(ROOT)),additional_training_limit_cu=control['additional_training_limit_cu'],
            resume_steps=control['resume_steps'],actual_new_update_verified=live.get('actual_new_update_verified',False),
            test_after_plateau_authorized=control.get('test_after_plateau_authorized',False),
            stopping_rule=control['stopping_rule'],prior_epoch2_and_epoch5_tests_preserved=True)
        output['v4_plateau'].update(converged=live.get('converged',False),completed_epochs=live.get('completed_epochs'),best_development_epoch=live.get('best_development_epoch'),full_test_status=plateau_test_status,full_test_complete=plateau_test_status=='complete')
        if plateau_test_status and live.get('converged'):
            output['v4_plateau']['status']='complete' if plateau_test_status=='complete' else 'converged_test_'+plateau_test_status
    atomic_json(folder/'inventory.json',output);print(json.dumps({k:output[k] for k in ['proposal_required_complete','annotation_progress','v3','budget']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
