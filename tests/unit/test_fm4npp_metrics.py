import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from pimm.engines.metrics import (
    fm4npp_event_ari,
)


def test_event_ari_keeps_shared_negative_prediction_label():
    truth = np.array([0, 0, 0, 0])
    prediction = np.array([-1, -1, -1, -1])

    assert fm4npp_event_ari(truth, prediction) == 1.0


def test_event_ari_scores_negative_truth_labels_instead_of_filtering_them():
    truth = np.array([0, 0, 1, 1, -1, -1])
    prediction = np.array([0, 0, 1, 1, 0, 1])

    score = fm4npp_event_ari(truth, prediction)

    assert score == adjusted_rand_score(truth, prediction)
    assert score < adjusted_rand_score(truth[:4], prediction[:4])


def test_event_ari_requires_aligned_nonempty_events():
    with pytest.raises(ValueError, match="same number"):
        fm4npp_event_ari([0, 1], [0])
    with pytest.raises(ValueError, match="at least one point"):
        fm4npp_event_ari([], [])
