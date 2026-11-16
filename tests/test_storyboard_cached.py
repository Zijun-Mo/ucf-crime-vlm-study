import pytest
import torch
from openjev_dss.storyboard_cached import PairFeatureView


def test_cached_pairs_preserve_window_overlap_and_last_frame_padding(tmp_path):
    blocks=torch.arange(11).reshape(11,1,1).expand(11,49,2048).to(torch.bfloat16).contiguous()
    path=tmp_path/'pairs.pt';torch.save(dict(signature={'source':'fixed'},features=blocks),path)
    view=PairFeatureView(path,{'source':'fixed'},10)
    for first in [0,4,8]:
        actual=view[first:first+4]
        for j in range(len(actual)):
            expected=torch.cat([blocks[min(first+j+k,10)] for k in range(4)])
            assert torch.equal(actual[j],expected)
    with pytest.raises(AssertionError):PairFeatureView(path,{'source':'different'},10)
