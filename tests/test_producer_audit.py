import ast
from pathlib import Path
import pytest
from openjev_dss.producer_audit import scoring_loop_signature

ROOT=Path(__file__).resolve().parents[1]


def test_formal_checkpoint_selection_keeps_the_frozen_full_test_scoring_loop():
    original=(ROOT/'research/run_inference_group.py').read_text()
    formal=(ROOT/'research/run_formal_inference.py').read_text()
    assert scoring_loop_signature(original)==scoring_loop_signature(formal)
    changed=formal.replace('score_windows(runtime.forward, features, 4)', 'score_windows(runtime.forward, features, 8)')
    assert scoring_loop_signature(original)!=scoring_loop_signature(changed)


def test_scoring_loop_audit_rejects_an_ambiguous_or_missing_producer():
    with pytest.raises(AssertionError,match='ambiguous'):
        scoring_loop_signature('for i, record in enumerate(records):\n pass\n'*2)
    with pytest.raises(AssertionError,match='Missing'):
        scoring_loop_signature('pass')
