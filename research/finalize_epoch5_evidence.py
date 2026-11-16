"""Audit settled epoch5 artifacts and close the authorized research event (CPU only)."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev_dss.followup_budget import amount, spending
from openjev_dss.learning import atomic_json


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    atomic_json(ROOT / path, value)


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base = 'research/localization_v4_epoch5_test'
    dest = 'results/qwen_localization_v4_epoch5_test'
    receipt_path = base + '/20261115T093601Z/receipt.json'
    receipt = read(receipt_path)
    final = read(dest + '/summary.json')
    original = read(dest + '/summary.autofinalized.original.json')
    assert receipt['status'] == final['status'] == 'complete'
    assert receipt['gpu'] == 'L4' and receipt['optimizer_updates'] == final['optimizer_updates'] == 0
    assert final['selected_epoch'] == 5 and final['configuration_selected_on_test'] is False
    assert final['gpu_released'] and receipt['release_observed_utc']
    # Only timestamps and cost metadata may change during offline recomputation.
    for key in ['metrics', 'paired_contrasts', 'reports', 'bootstrap']:
        assert original[key] == final[key], key
    write(base + '/offline_recomputation_audit.json', dict(
        passed=True, verified_utc=now, metrics_and_bootstrap_exact_match=True,
        original_summary_sha256=digest(ROOT / dest / 'summary.autofinalized.original.json'),
        offline_summary_sha256=digest(ROOT / dest / 'summary.json'),
        command='PYTHONPATH=src .venv/bin/python scripts/finalize_localization_test.py ' + receipt_path + ' --offline',
        gpu_allocations=0, network_required=False))

    audit = read(base + '/complete_prediction_audit.json')
    hashes = read(dest + '/prediction_sha256.json')
    assert audit['passed'] and (audit['videos'], audit['windows'], audit['frames']) == (290, 37188, 1111808)
    assert audit['optimizer_updates'] == 0 and len(hashes) == 290
    for relative, expected in hashes.items():
        path = ROOT / dest / relative
        assert digest(path) == expected == audit['prediction_sha256'][path.name]
    native = read('results/study_formal_v2/qwen_zs_p1_4s.json')['metrics']
    assert all(abs(v - native[k]['value']) < 1e-12 for k, v in final['metrics']['baseline'].items())
    stop = read('research/localization_v4_continuation/operator_user_stop.json')
    assert stop['complete'] and stop['selected_checkpoint_remote_verified'] and stop['resume_remote_verified']
    assert stop['committed_steps'] == 806 and set(stop['optimizer_steps']) == {806}
    assert stop['selected_file_sha256'] == audit['checkpoint_file_sha256']
    assert stop['selected_tensor_sha256'] == final['checkpoint_tensor_sha256'] == audit['checkpoint_tensor_sha256']
    assert stop['unvalidated_epoch6_updates'] == 806 - 785 == 21
    assert read('research/localization_v4_continuation/control.json')['authorized'] is False

    transfer = read('results/future_work_v1/frame_transfer_analysis.json')
    contributions = read('results/future_work_v1/auc_contributions.json')['contributions']
    for arm, row in contributions.items():
        delta = 100 * (transfer['results'][arm]['pooled_auroc'] - transfer['results']['zero_shot']['pooled_auroc'])
        assert abs(row['normal_video_comparison_pp'] + row['abnormal_background_comparison_pp'] - delta) < 1e-10
        assert abs(row['total_pooled_auroc_change_pp'] - delta) < 1e-10
    per_video = read('results/future_work_v1/per_video_transfer.json')
    assert len(per_video) == 140
    for field, groups in transfer['stratified_within_differences'].items():
        assert sum(row['videos'] for row in groups.values()) == 140
        for label, row in groups.items():
            rows = [v for v in per_video if v[field] == label]
            assert len(rows) == row['videos']
            for arm, mean in row['mean_within_deltas'].items():
                assert abs(sum(v['within_deltas'][arm] for v in rows) / len(rows) - mean) < 1e-12
    ordered = sorted(per_video, key=lambda r: (r['within_deltas']['minmax_soft_epoch5'], r['name']))
    cases = read(dest + '/case_selection.json')['cases']
    for role, row in [('worst', ordered[0]), ('median', ordered[len(ordered)//2]), ('best', ordered[-1])]:
        assert cases[role]['name'] == row['name']
    manifest = read(dest + '/case_frames/manifest.json')
    assert len(manifest['cases']) == 3
    for row in manifest['cases']:
        assert digest(ROOT / dest / 'case_frames' / row['image']) == row['image_sha256']
        assert row['name'] == cases[row['role']]['name']
    assert read(base + '/ai_use_and_review.json')['case_images_reviewed'] == 3
    contract = read('research/future_work_v1/course_contract.json')
    assert digest(Path(contract['submitted_proposal'])) == contract['proposal_sha256']
    assert digest(Path(contract['course_archive'])) == contract['course_archive_sha256']
    assert (ROOT / 'plan.md').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:plan.md'], cwd=ROOT)
    inventory = read('results/future_work_v1/inventory.json')
    assert len(inventory['conditions']) == 17 and all(c['status'] == 'complete' for c in inventory['conditions'])
    assert inventory['proposal_required_complete'] == 13

    usage = subprocess.check_output(['colab', 'usage'], text=True)
    sessions = subprocess.check_output(['colab', 'sessions'], text=True)
    assert 'Active assignments: 0' in usage and 'Usage rate: 0.00/hr' in usage
    assert 'No active sessions' in sessions
    budget = spending(ROOT)
    assert budget['total_estimate_cu'] < 80 and amount(receipt) < 3
    assert budget['phases']['annotation_cpu'] < .5
    archive = ROOT / 'research/future_work_v1/history/before_epoch5_final'
    for relative in ['research/future_work_v1/final_acceptance.json', 'research/future_work_v1/completion.json',
                     'research/future_work_v1/budget.json', 'research/future_work_v1/release_audit.json']:
        target = archive / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)

    release = dict(verified_utc=now, gpu_released=True, active_assignments=0, usage_rate_cu_per_hour=0,
                   usage=usage, sessions=sessions, cpu_case_allocations_also_released=True,
                   test_receipt=receipt_path, training_receipt='research/runs/localization_train_20261115T030626Z/receipt.json')
    write(base + '/release_audit.json', release)
    write('research/future_work_v1/release_audit.json', release)
    for relative in ['research/localization_v4_continuation/result_summary.json',
                     'research/localization_v4_continuation/launch.json',
                     'research/localization_v4_continuation/handoff_state.json']:
        state = read(relative)
        state.update(status='user_stopped_epoch5_test_complete', updated_utc=now, selected_epoch=5,
                     selected_optimizer_updates=785, actual_committed_steps=806, verified_current_steps=806,
                     unvalidated_epoch6_updates_excluded=21, converged=False, training_stopped_by_user=True,
                     full_test_complete=True, full_test_summary=dest + '/summary.json',
                     active_gpu_assignments=0, gpu_released=True, monitor_running=False,
                     event_monitor_only=False, budget=budget,
                     additional_training_used_cu=budget['phases']['v3_training'] - 14.700343217544445,
                     additional_training_remaining_cu=0., unused_training_allowance_closed=True)
        state['checkpoint'] = state['selected_checkpoint']
        write(relative, state)
    for relative in [base + '/launch.json', base + '/handoff_state.json']:
        state = read(relative)
        state.update(status='complete', updated_utc=now, completed_videos=290,
                     last_observed_completed_videos=290, scored_windows=37188, frames=1111808,
                     optimizer_updates=0, training=False, gpu_released=True, active_gpu_assignments=0,
                     monitor_running=False, summary=dest + '/summary.json', budget=budget)
        write(relative, state)
    state = read('research/future_work_v1/budget.json')
    state.update(status='complete_training_stopped_by_user', updated_utc=now,
                 remaining_cu=budget['remaining_cu'], current_spending=budget)
    write('research/future_work_v1/budget.json', state)
    acceptance = read('research/future_work_v1/final_acceptance.json')
    acceptance.pop('unvalidated_epoch3_steps_excluded', None)
    acceptance.update(verified_utc=now, selected_epoch=5, selected_optimizer_updates=785,
        total_committed_updates=806, unvalidated_epoch6_steps_excluded=21,
        checkpoint_file_sha256=audit['checkpoint_file_sha256'], checkpoint_tensor_sha256=audit['checkpoint_tensor_sha256'],
        additional_complete_tests=['storyboard_context','teacher_constraint_epoch1','minmax_soft_epoch2','minmax_soft_epoch5'],
        total_formal_conditions=17, v4_status='five completed epochs;user stop after epochs4and5 within-video decline;frozen epoch5 complete test',
        converged=False, stop_rule_added_after_observing_development=True,
        actual_update_audit='research/localization_v4_continuation/operator_user_stop.json',
        budget=budget, current_targeted_tests=dict(passed=31, command='PYTHONPATH=src .venv/bin/python -m pytest tests/test_frame_transfer.py tests/test_followup_test.py tests/test_followup_budget.py tests/test_colab_kernel_initialization.py tests/test_colab_one_shot_control.py tests/test_trial_events.py -q'),
        acceptance_covers='13 proposal conditions plus montage,teacher,E2 and selectedE5 full tests;no convergence claim',
        offline_audit=base+'/offline_recomputation_audit.json', formal_test_summary=dest+'/summary.json',
        direct_paired_comparisons=dest+'/method_comparisons.json', ai_review=base+'/ai_use_and_review.json',
        active_gpu_assignments=0, training_authorized=False, monitor_running=False)
    acceptance.pop('new_authorized_continuation', None)
    write('research/future_work_v1/final_acceptance.json', acceptance)
    write('research/future_work_v1/completion.json', dict(status='complete', finished_utc=now,
        required_conditions_completed=13, total_formal_conditions=17, formal_teacher_test='complete',
        new_development_annotations='12 assistant annotated/reviewed;34 total anomaly timelines',
        v3_training='not started:historical quality gate failure retained',
        v4_training='five complete epochs;785 selected updates;806 committed;21 unvalidated excluded;user stop',
        converged=False, v4_epoch2_full_test='complete and preserved', v4_epoch5_full_test='complete:290videos/37188windows/1111808frames',
        selected_epoch=5, analysis_and_handoff='results/future_work_v1/README.md', gpu_active_assignments=0,
        unspent_phase_budget_cu=budget['remaining_cu'], routine_model_polling=False, event_monitor='retired:all necessary events handled',
        acceptance='research/future_work_v1/final_acceptance.json', remaining_authorized_gpu_work=False))
    # Retire only our identified completed-test monitor, after persisting evidence.
    event_path = base + '/monitor/events/a0559e9921a8d8fd89693895.json'
    event = read(event_path)
    event.update(handled_utc=now, action_summary='Verified290/37188/1111808,zero test updates,epoch5 checkpoint hashes,offline exact recomputation,direct bootstrap,case evidence,budget and GPU release;published completed handoff;no training resumed',
                 acceptance='research/future_work_v1/final_acceptance.json')
    write(event_path, event)
    monitor_path = base + '/monitor/state.json'
    monitor = read(monitor_path)
    pid = monitor.get('pid')
    proc = Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        command = proc.read_bytes().replace(b'\0', b' ').decode()
        assert 'scripts/monitor_localization_test.py' in command and 'localization_v4_epoch5_test' in command
        os.kill(pid, signal.SIGTERM)
    monitor.update(status='settled', lifecycle='retired', retired_utc=now, monitor_running=False, budget=budget)
    monitor['delivered'][event['id']] = 'handled'
    write(monitor_path, monitor)
    print(json.dumps(dict(passed=True, selected_epoch=5, formal_conditions=17, budget=budget,
                          test_cost_cu=amount(receipt), active_assignments=0), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
