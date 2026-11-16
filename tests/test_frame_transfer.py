import numpy as np
from scripts.analyze_frame_transfer import decompose


def test_known_pairs_separate_within_video_from_cross_video_detection():
    labels=[np.array([0,1,0,1]),np.array([0,0])]
    scores=[np.array([0.,3.,1.,2.]),np.array([4.,-1.])]
    result=decompose(labels,scores,[False,True])
    assert result['pooled_auroc']==.75
    assert result['same_video_pair_weight']==.5
    assert result['same_video_pair_weighted_auroc']==1.
    assert result['cross_video_pair_auroc']==.5
    assert result['anomaly_vs_normal_video_auroc']==.5
    assert result['anomaly_vs_abnormal_background_auroc']==1.
