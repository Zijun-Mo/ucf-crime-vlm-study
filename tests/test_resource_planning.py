import copy
import pytest
from openjev_dss.resource_planning import choose_scale_hardware, estimate_scale_cost, budgeted_runtime_minutes


def candidate(gpu, rate, throughput):
    return ({'status':'passed','test_labels_used':False,'dtype':'bf16','batch':4,'prefix_cache':False,
             'gpu_gib':24.,'peak_gpu_gib':20.,'language_windows_s':throughput,'vision_batch4_s':[.5,.25],
             'job':{'duration_s':4},'pair_feature_gate':{'allowed':True},'elapsed_s':180.},
            {'gpu':gpu,'usage_allocated':f'Usage rate: {rate}/hr','receipt_path':gpu})


CONFIG={'text_config':{'hidden_size':4096}}
UPLOAD={'source_bytes':30_000_000,'trials':[{'mode':'single_request','exit_code':0,'seconds':5.}]}


def test_high_rate_full_job_is_capped_while_short_probe_keeps_its_bound():
    assert budgeted_runtime_minutes(46.64, 10, 360) == 267
    assert budgeted_runtime_minutes(46.64, 10, 15) == 15
    assert budgeted_runtime_minutes(46.64, 1.94, 360) == 360
    with pytest.raises(ValueError,match='do not allocate'):
        budgeted_runtime_minutes(2.1, 10, 360)
    for rate in [0, -1, float('nan')]:
        with pytest.raises(ValueError):
            budgeted_runtime_minutes(46.64, rate, 360)


def test_faster_accelerator_can_cost_more_when_io_dominates():
    slow=candidate('L4',2.,10.)
    fast=candidate('A100',6.,30.)
    result=choose_scale_hardware([slow,fast],CONFIG,UPLOAD)
    assert result['selected_gpu']=='L4'
    assert not result['test_scores_used']
    assert len(result['candidates'])==2


def test_oom_probe_is_rejected_and_remaining_candidate_is_used():
    bad,receipt=candidate('L4',2.,10.)
    bad['status']='error'
    good=candidate('A100',5.,30.)
    result=choose_scale_hardware([(bad,receipt),good],CONFIG,UPLOAD)
    assert result['selected_gpu']=='A100' and len(result['rejected'])==1
    good[0]['test_labels_used']=True
    with pytest.raises(ValueError,match='No measured'):
        choose_scale_hardware([(bad,receipt),good],CONFIG,UPLOAD)


def test_pair_gate_failure_expands_both_vision_work_and_feature_upload():
    probe,receipt=candidate('L4',2.,10.)
    pair=estimate_scale_cost(probe,receipt,CONFIG,UPLOAD)
    probe['pair_feature_gate']['allowed']=False
    full=estimate_scale_cost(probe,receipt,CONFIG,UPLOAD)
    assert full['feature_bytes_estimate']>pair['feature_bytes_estimate']*3.9
    assert full['components']['vision_s']>pair['components']['vision_s']*3.9
    assert full['estimated_cu']>pair['estimated_cu']


def test_candidate_specific_network_measurement_can_change_hardware_choice():
    a=candidate('L4',2.,10.)
    b=candidate('A100',2.,30.)
    assert choose_scale_hardware([a,b],CONFIG,UPLOAD)['selected_gpu']=='A100'
    b[0]['upload_benchmark']={'source_bytes':30_000_000,'trials':[{'mode':'single_request','exit_code':0,'seconds':50.}]}
    assert choose_scale_hardware([a,b],CONFIG,UPLOAD)['selected_gpu']=='L4'
