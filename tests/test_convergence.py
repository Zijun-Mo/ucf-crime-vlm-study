import copy
import pytest
from openjev_dss.convergence import record_validation, convergence_summary

POLICY={'min_delta':.002,'patience':3,'min_epochs':3}


def test_plateau_is_demonstrated_by_multiple_full_validation_epochs():
    state={}
    results=[record_validation(state,i,s,POLICY) for i,s in enumerate([.8,.8,.799,.8],1)]
    assert results[-1][1] and state['stopped_epoch']==4
    assert state['best_epoch']==1 and state['stop_reason']=='validation_plateau'


def test_small_gains_select_the_best_checkpoint_without_resetting_patience():
    state={}
    for i,s in enumerate([.8,.8005,.801,.8015],1):record_validation(state,i,s,POLICY)
    assert state['best_epoch']==4 and state['bad_epochs']==3
    assert state['stop_reason']=='validation_plateau'


def test_epoch_limit_does_not_prove_convergence_and_can_be_extended():
    state={}
    for i,s in enumerate([.70,.72,.74],1):record_validation(state,i,s,POLICY)
    assert convergence_summary({'qwen':state},3)=='needs_extension'
    assert convergence_summary({'qwen':state},5)=='running'
    for i,s in enumerate([.74,.74,.74],4):record_validation(state,i,s,POLICY)
    assert convergence_summary({'qwen':state},6)=='complete'


def test_resume_preserves_patience_and_best_checkpoint():
    direct={};resumed={}
    scores=[.7,.72,.721,.719,.7215]
    for i,s in enumerate(scores,1):record_validation(direct,i,s,POLICY)
    for i,s in enumerate(scores[:3],1):record_validation(resumed,i,s,POLICY)
    resumed=copy.deepcopy(resumed)
    for i,s in enumerate(scores[3:],4):record_validation(resumed,i,s,POLICY)
    assert direct==resumed
    with pytest.raises(ValueError,match='contiguous'):record_validation(resumed,5,.8,POLICY)
    with pytest.raises(ValueError,match='finite'):record_validation({},1,float('nan'),POLICY)
