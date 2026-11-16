import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from openjev_dss.study_evaluation import frame_labels, frame_scores, histogram_bootstrap


def test_cluster_histograms_match_explicit_resampling_with_ties():
    labels = [np.array([0, 1, 1]), np.array([0, 0]), np.array([1, 0, 1, 0])]
    scores = [np.array([.4, .4, .7]), np.array([.1, .4]), np.array([.7, .4, .1, .1])]
    weights = np.array([[1, 1, 1], [2, 0, 1], [0, 2, 2], [1, 0, 0]])
    point, samples = histogram_bootstrap(labels, scores, weights, batch=2)
    y, z = np.concatenate(labels), np.concatenate(scores)
    np.testing.assert_allclose(point, [roc_auc_score(y, z), average_precision_score(y, z)])
    for multiplicities, actual in zip(weights, samples):
        expanded_y = np.concatenate([np.tile(y, n) for y, n in zip(labels, multiplicities)])
        expanded_z = np.concatenate([np.tile(z, n) for z, n in zip(scores, multiplicities)])
        expected = [roc_auc_score(expanded_y, expanded_z), average_precision_score(expanded_y, expanded_z)]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_original_frame_boundaries_and_stride_are_preserved():
    r = {'n_frames': 7, 'fps': 2, 'intervals': [[2, 3], [7, 99], [-1, -1]]}
    np.testing.assert_array_equal(frame_labels(r), [0, 1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(frame_scores(r, [0, 1, 2, 3], [0., 2., 4., 6.], 2, 1), [0, 0, 1, 1, 3, 3, 5])
    np.testing.assert_array_equal(frame_scores(r, [0, 1, 2, 3], [0., 2., 4., 6.], 2, 2), [0, 0, 0, 0, 4, 4, 4])
