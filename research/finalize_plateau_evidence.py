"""Audit settled epoch9 results and close the authorized training/test event."""
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.followup_budget import amount,spending
from openjev_dss.learning import atomic_json


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):atomic_json(ROOT/name,data)


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    base='research/localization_v4_plateau_test';dest='results/qwen_localization_v4_plateau_test'
    receipt_path=base+'/20261115T205551Z/receipt.json';r=read(receipt_path);summary=read(dest+'/summary.json')
    selected=read('research/localization_v4_plateau/selected_checkpoint.json');train=read('research/localization_v4_plateau/complete_training_acceptance.json')
    registration=read(base+'/registration.json');policy=read(base+'/policy.json')
    assert r['status']==summary['status']=='complete' and r['optimizer_updates']==summary['optimizer_updates']==0
    assert selected['selection_frozen'] and selected['selected_epoch']==summary['selected_epoch']==policy['selected_epoch']==9
    assert datetime.datetime.fromisoformat(selected['selection_frozen_utc'])<datetime.datetime.fromisoformat(r['allocated_utc'])
    assert not summary['configuration_selected_on_test'] and selected['test_used_for_selection'] is False
    assert sha(ROOT/base/'policy.json')==registration['policy_sha256']
    original=read(dest+'/summary.autofinalized.original.json')
    for k in ['metrics','paired_contrasts','reports','bootstrap']:assert original[k]==summary[k]
    gate=read(base+'/complete_prediction_audit.json');hashes=read(dest+'/prediction_sha256.json')
    assert gate['passed'] and (gate['videos'],gate['windows'],gate['frames'])==(290,37188,1111808)
    assert gate['optimizer_updates']==0 and len(hashes)==290
    for name,expected in hashes.items():
        path=ROOT/dest/name;assert sha(path)==expected==gate['prediction_sha256'][path.name]
    for k in ['checkpoint_file_sha256','checkpoint_tensor_sha256']:assert gate[k]==selected[k]==policy[k]
    assert summary['checkpoint_tensor_sha256']==selected['checkpoint_tensor_sha256']
    official=read('results/study_formal_v2/qwen_zs_p1_4s.json')['metrics']
    for k,v in summary['metrics']['baseline'].items():assert abs(v-official[k]['value'])<1e-12
    baseline_hashes=read(base+'/offline_preallocation_gate.json')['baseline_prediction_sha256']
    for name,expected in baseline_hashes.items():assert sha(ROOT/policy['baseline_source']/(name+'.json'))==expected
    assert r['gpu']=='L4' and summary['gpu_released'] and r['release_observed_utc']
    with zipfile.ZipFile(ROOT/r['source_archive']) as archive:
        assert sha(ROOT/r['source_archive'])==r['source_sha256']
        assert hashlib.sha256(archive.read('research/run_localization_test.py')).hexdigest()==sha(ROOT/'research/run_localization_test.py')
        archived_policy=archive.read(base+'/policy.json');local_policy=(ROOT/base/'policy.json').read_bytes()
        # The deployment serializer appends a newline; the frozen local JSON does not.
        assert archived_policy.rstrip(b'\n')==local_policy.rstrip(b'\n') and json.loads(archived_policy)==policy
        write(base+'/deployed_policy_audit.json',dict(passed=True,archive_sha256=hashlib.sha256(archived_policy).hexdigest(),
            frozen_local_sha256=registration['policy_sha256'],json_exactly_equal=True,only_difference='terminal newline'))
    first=read(base+'/first_prediction_audit.json')
    assert first['passed'] and first['selected_epoch']==9 and first['optimizer_updates']==0
    assert first['checkpoint']['tensor_sha256']==selected['checkpoint_tensor_sha256']
    assert train['passed'] and train['completed_epochs']==9 and train['optimizer_updates']==1413 and train['unvalidated_tail_updates']==0
    replay=read('research/localization_v4_plateau/stopping_rule_replay.json')
    assert replay['passed'] and replay['plateau_reached'] and replay['bad_epochs']==2 and replay['best_epoch']==9
    assert not read('research/localization_v4_plateau/control.json')['authorized']
    transfer=read('results/future_work_v1/frame_transfer_analysis.json');contributions=read('results/future_work_v1/auc_contributions.json')['contributions']
    for arm,row in contributions.items():
        delta=100*(transfer['results'][arm]['pooled_auroc']-transfer['results']['zero_shot']['pooled_auroc'])
        assert abs(row['normal_video_comparison_pp']+row['abnormal_background_comparison_pp']-delta)<1e-10
    pv=read('results/future_work_v1/per_video_transfer.json');assert len(pv)==140
    for field,groups in transfer['stratified_within_differences'].items():
        assert sum(v['videos'] for v in groups.values())==140
        for name,row in groups.items():
            values=[v for v in pv if v[field]==name];assert len(values)==row['videos']
            for arm,point in row['mean_within_deltas'].items():assert abs(sum(v['within_deltas'][arm] for v in values)/len(values)-point)<1e-12
    cases=read(dest+'/case_selection.json')['cases'];ordered=sorted(pv,key=lambda v:(v['within_deltas']['minmax_soft_epoch9'],v['name']))
    for role,row in [('worst',ordered[0]),('median',ordered[len(ordered)//2]),('best',ordered[-1])]:assert cases[role]['name']==row['name']
    manifest=read(dest+'/case_frames/manifest.json');assert len(manifest['cases'])==3
    records={v['name']:v for v in policy['records']}
    for row in manifest['cases']:
        assert sha(ROOT/dest/'case_frames'/row['image'])==row['image_sha256']
        assert row['name']==cases[row['role']]['name']
        assert row['source_sha256']==records[row['name']]['sha256']
        assert row['canonical_sha256']==policy['metadata'][row['name']]['canonical_sha256']
    assert read(base+'/ai_use_and_review.json')['case_images_reviewed']==3
    contract=read('research/future_work_v1/course_contract.json')
    assert sha(contract['submitted_proposal'])==contract['proposal_sha256'] and sha(contract['course_archive'])==contract['course_archive_sha256']
    assert (ROOT/'plan.md').read_bytes()==subprocess.check_output(['git','show','HEAD:plan.md'],cwd=ROOT)
    usage=subprocess.check_output(['colab','usage'],text=True);sessions=subprocess.check_output(['colab','sessions'],text=True)
    assert 'Active assignments: 0' in usage and 'Usage rate: 0.00/hr' in usage and 'No active sessions' in sessions
    budget=spending(ROOT);assert budget['total_estimate_cu']<100 and amount(r)<3 and train['additional_training_used_cu']<20 and budget['phases']['annotation_cpu']<.5
    history=ROOT/'research/future_work_v1/history/before_epoch9_final'
    for name in ['final_acceptance.json','completion.json','budget.json','release_audit.json']:
        target=history/'research/future_work_v1'/name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'research/future_work_v1'/name,target)
    release=dict(verified_utc=now,gpu_released=True,active_assignments=0,usage_rate_cu_per_hour=0,usage=usage,sessions=sessions,
                 training_receipt='research/runs/localization_train_20261115T134602Z/receipt.json',test_receipt=receipt_path,cpu_case_preparation_also_released=True)
    write(base+'/release_audit.json',release);write('research/future_work_v1/release_audit.json',release)
    for name in [base+'/launch.json',base+'/handoff_state.json']:
        state=read(name);state.update(status='complete',updated_utc=now,selected_epoch=9,completed_videos=290,last_observed_completed_videos=290,
            scored_windows=37188,frames=1111808,optimizer_updates=0,training=False,test_scoring_verified=True,gpu_released=True,active_gpu_assignments=0,
            monitor_running=False,summary=dest+'/summary.json',budget=budget);write(name,state)
    for name in ['result_summary.json','handoff_state.json','launch.json']:
        path='research/localization_v4_plateau/'+name;state=read(path);state.update(status='complete',updated_utc=now,completed_epochs=9,current_epoch=None,
            selected_epoch=9,selected_optimizer_updates=1413,observed_steps=1413,converged=True,bad_epochs=2,full_test_complete=True,
            full_test_summary=dest+'/summary.json',gpu_released=True,active_gpu_assignments=0,monitor_running=False,budget=budget);write(path,state)
    b=read('research/future_work_v1/budget.json');b.update(status='complete',updated_utc=now,current_spending=budget,remaining_cu=budget['remaining_cu']);write('research/future_work_v1/budget.json',b)
    checks=read(base+'/current_targeted_checks.json')
    acceptance=dict(status='passed',verified_utc=now,required_proposal_conditions=13,required_conditions_completed=13,total_formal_conditions=18,
        additional_complete_tests=['montage','teacher_constraint_epoch1','minmax_soft_epoch2','minmax_soft_epoch5','minmax_soft_plateau_epoch9'],
        videos=290,windows=37188,frames=1111808,selected_epoch=9,selected_optimizer_updates=1413,total_committed_updates=1413,unvalidated_tail_updates=0,
        actual_parameter_updates_verified=True,actual_update_audit='research/localization_v4_plateau/actual_update_acceptance.json',
        checkpoint_file_sha256=selected['checkpoint_file_sha256'],checkpoint_tensor_sha256=selected['checkpoint_tensor_sha256'],
        all_prediction_hashes_verified=True,baseline_all_metrics_match_accepted_source=True,baseline_reused_without_new_inference=True,
        test_used_for_selection=False,formal_test_optimizer_updates=0,bootstrap=summary['bootstrap'],
        converged=True,convergence_definition='Configured development plateau;bad_epochs2;not mathematical convergence proof',
        training_acceptance='research/localization_v4_plateau/complete_training_acceptance.json',selected_checkpoint='research/localization_v4_plateau/selected_checkpoint.json',
        offline_audit=base+'/offline_recomputation_audit.json',formal_test_summary=dest+'/summary.json',direct_paired_comparisons=dest+'/method_comparisons.json',
        decomposition_verified=True,subgroup_counts_verified=True,fixed_case_rule_verified=True,case_image_hashes_verified=3,case_contact_sheets_reviewed=3,
        approximate_assistant_reviewed_development_anomalies=34,new_reviewed_anomalies=12,normal_frame_dev_videos=78,
        proposal_unchanged=True,plan_md_matches_git_head=True,budget=budget,last_additional_training_used_cu=train['additional_training_used_cu'],last_test_cu=amount(r),
        current_targeted_tests=checks,gpu_released=True,active_gpu_assignments=0,monitor_running=False,training_authorized=False,
        resource_audit='research/future_work_v1/release_audit.json',ai_review=base+'/ai_use_and_review.json',
        localization_noninferiority_proven=False,extra_training_main_auroc_interval_positive=True,
        prior_epoch2_and_epoch5_tests_preserved=True,acceptance_covers='18completedformalconditions,plateau training,selectedepoch9test and analysis;no pendingGPU work')
    write('research/future_work_v1/final_acceptance.json',acceptance)
    write('research/future_work_v1/completion.json',dict(status='complete',finished_utc=now,required_conditions_completed=13,total_formal_conditions=18,
        formal_teacher_test='complete',new_development_annotations='12newreviewed;34assistant-reviewed approximate anomaly timelines total',
        v3_training='notstarted:historical hard-candidate qualitygate failure retained',v4_training='ninecompletedepochs/1413updates;configureddevelopmentplateau reached',
        converged=True,selected_epoch=9,selected_optimizer_updates=1413,original_epoch2_and_epoch5_tests_preserved=True,
        plateau_epoch9_full_test='complete290videos/37188windows/1111808frames',analysis_and_handoff='results/future_work_v1/README.md',
        gpu_active_assignments=0,cpu_active_assignments=0,routine_model_polling=False,event_monitor='retired after completed event',
        remaining_authorized_gpu_work=False,unspent_phase_budget_cu=budget['remaining_cu'],acceptance='research/future_work_v1/final_acceptance.json'))
    event_path=base+'/monitor/events/00da586c436671471bf5b9f3.json';event=read(event_path)
    event.update(handled_utc=now,action_summary='Verifiedselected9and1413updates,all290predictions/SHA/37188windows/1111808frames,zerotestupdates,offlineexactrecomputationand1000pairedvideo intervals;primaryAUROCbenefitvsnative,E5andoldMIL;localizationnoninferioritynotproven;updated18condition evidence,handoff,cases,costs andconfirmedallGPU/CPUsreleased.No furthertrainingorcheckpointtests.',acceptance='research/future_work_v1/final_acceptance.json');write(event_path,event)
    state=read(base+'/monitor/state.json');pid=state['pid'];proc=Path('/proc')/str(pid)/'cmdline'
    if proc.exists():
        cmd=proc.read_bytes();assert b'scripts/monitor_localization_test.py' in cmd and b'localization_v4_plateau_test' in cmd
        os.kill(pid,signal.SIGTERM)
    state.update(status='settled',lifecycle='retired',monitor_running=False,retired_utc=now,budget=budget);state['delivered'][event['id']]='handled';write(base+'/monitor/state.json',state)
    subprocess.run([sys.executable,'scripts/refresh_experiment_inventory.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    index=read('results/future_work_v1/inventory.json');assert len(index['conditions'])==18 and all(r['status']=='complete' for r in index['conditions'])
    print(json.dumps(dict(passed=True,formal_conditions=18,selected_epoch=9,test_cu=amount(r),budget=budget,active_assignments=0),indent=2))


if __name__=='__main__':main()
