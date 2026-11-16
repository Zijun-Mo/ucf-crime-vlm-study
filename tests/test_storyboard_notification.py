import pytest
from openjev_dss.storyboard_notification import completion_prompt


@pytest.mark.parametrize('formal,status',[(True,'complete'),(True,'error'),(False,'complete'),(False,'error')])
def test_completion_payload_is_constructed_for_success_and_failure(formal,status):
    result=completion_prompt(dict(status=status,formal_test=formal),'event.json','receipt.json')
    assert isinstance(result,str) and 'event.json' in result and 'receipt.json' in result
    assert status in result and 'L4' in result
    if formal:
        assert '完整290视频' in result and '不要补跑仅文字组' in result and '11CU' in result
    else:
        assert '44视频开发对照' in result and '3CU' in result
