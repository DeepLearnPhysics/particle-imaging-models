"""FM4NPP adjusted-Rand evaluation on every event point."""

from __future__ import annotations

import numpy as np


# FM4NPP/PP_collision (arXiv:2508.14087): event-wise ARI over all points.
def fm4npp_event_ari(truth: object, prediction: object) -> float:
    """Compute sklearn ARI on every point in one event without label rewriting."""

    from sklearn.metrics import adjusted_rand_score

    truth_array = np.asarray(truth).reshape(-1)
    prediction_array = np.asarray(prediction).reshape(-1)
    if truth_array.shape != prediction_array.shape:
        raise ValueError(
            "truth and prediction must contain the same number of event points, got "
            f"{truth_array.size} and {prediction_array.size}"
        )
    if truth_array.size == 0:
        raise ValueError("FM4NPP event ARI requires at least one point")
    return float(adjusted_rand_score(truth_array, prediction_array))


__all__ = ["fm4npp_event_ari"]
