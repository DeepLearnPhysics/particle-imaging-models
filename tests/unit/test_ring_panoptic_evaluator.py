"""CPU tests for ring evaluator target adaptation and regression metrics."""

import numpy as np
import torch

from pimm.engines.hooks.eval.rings import (
    RingPanopticEvaluator,
    _log_fractional_error,
    match_rings,
)


def test_split_overlap_targets_uses_event_offsets_and_valid_rings():
    input_dict = {
        "offset": torch.tensor([3, 5]),
        "inst_pe": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "ring_valid": torch.tensor([True, False, True, False, True, False]),
        "ring_pid": torch.tensor([2, -1, 0, -1, 3, -1]),
        "ring_energy": torch.arange(6, dtype=torch.float),
        "ring_momentum": torch.arange(6, dtype=torch.float) + 10,
        "ring_vertex": torch.arange(18, dtype=torch.float).reshape(6, 3),
        "ring_direction": torch.arange(18, dtype=torch.float).reshape(6, 3),
    }

    targets = RingPanopticEvaluator._split_overlap_targets(input_dict, 2)

    assert len(targets) == 2
    assert targets[0]["masks"].shape == (2, 3)
    assert targets[0]["masks"].tolist() == [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert targets[0]["pid"].tolist() == [2, 0]
    assert targets[1]["masks"].shape == (1, 2)
    assert targets[1]["masks"].tolist() == [[1.0, 1.0]]
    assert targets[1]["pid"].tolist() == [3]
    assert targets[1]["energy"].tolist() == [4.0]
    assert targets[1]["vertex"].shape == (1, 3)


def test_particle_regression_q68_is_scoped_to_true_electrons_and_muons():
    evaluator = RingPanopticEvaluator(
        coord_scale=10.0,
        particle_classes=(0, 2),
    )
    matched_pids = [0, 2, 1, 0, 2, 1, 0, 2]
    vertex_m = [1.0, 4.0, 100.0, 2.0, 5.0, 200.0, 3.0, 6.0]
    momentum_frac = [0.1, 0.4, 9.0, 0.2, 0.5, 8.0, 0.3, 0.6]

    metrics = evaluator._particle_regression_metrics(
        matched_pids,
        momentum_frac,
        vertex_m,
    )

    assert np.isclose(metrics["vertex_e_q68_m"], np.quantile([1, 2, 3], 0.68))
    assert np.isclose(metrics["vertex_e_q68_norm"], np.quantile([1, 2, 3], 0.68) / 10)
    assert np.isclose(metrics["vertex_mu_q68_m"], np.quantile([4, 5, 6], 0.68))
    assert np.isclose(
        metrics["momentum_e_frac_q68"], np.quantile([0.1, 0.2, 0.3], 0.68)
    )
    assert np.isclose(
        metrics["momentum_mu_frac_q68"], np.quantile([0.4, 0.5, 0.6], 0.68)
    )
    reordered = RingPanopticEvaluator(coord_scale=10.0, particle_classes=(2, 0))
    reordered_metrics = reordered._particle_regression_metrics(
        [2 if pid == 0 else 0 if pid == 2 else pid for pid in matched_pids],
        momentum_frac,
        vertex_m,
    )
    assert reordered_metrics == metrics


def test_log_fractional_error_is_absolute_delta_p_over_true_p():
    true_log = np.log(100.0)

    assert np.isclose(_log_fractional_error(np.log(80.0), true_log), 0.2)
    assert np.isclose(_log_fractional_error(np.log(120.0), true_log), 0.2)


def test_match_rings_gives_a_truth_ring_to_its_best_duplicate_in_any_query_order():
    truth = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    loose = torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.bool)  # IoU 3/5
    tight = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)  # IoU 1
    for pred, best in (
        (torch.stack([loose, tight]), 1),
        (torch.stack([tight, loose]), 0),
    ):
        rows, cols, _ = match_rings(pred, truth)
        assert rows.tolist() == [best] and cols.tolist() == [0]
    # the IoU bonus never costs a match: A fits ring 0 best but goes to ring 1 so that B can take ring 0
    site = torch.arange(12)
    truth = torch.stack([site < 4, site < 8])
    a, b = site < 4, (site < 4) | (site >= 8)  # A: IoU 1 and 1/2; B: IoU 1/2 and 1/3
    rows, cols, _ = match_rings(torch.stack([a, b]), truth)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]
