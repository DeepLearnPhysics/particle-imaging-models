"""Unit tests for multi-label instance checkpoint selection."""

import pytest

from pimm.engines.hooks.eval.instance import (
    InstanceSegmentationEvaluator,
)


def test_joint_selection_reductions():
    values = [0.8, 0.5]

    assert InstanceSegmentationEvaluator._reduce_selection_metrics(
        values, "mean"
    ) == pytest.approx(0.65)
    assert InstanceSegmentationEvaluator._reduce_selection_metrics(
        values, "harmonic_mean"
    ) == pytest.approx(2.0 / (1.0 / 0.8 + 1.0 / 0.5))


def test_harmonic_selection_requires_every_branch_to_be_nonzero():
    assert (
        InstanceSegmentationEvaluator._reduce_selection_metrics(
            [0.9, 0.0], "harmonic_mean"
        )
        == 0.0
    )


def test_invalid_selection_reduction_is_rejected():
    with pytest.raises(ValueError, match="selection_reduction"):
        InstanceSegmentationEvaluator(selection_reduction="maximum")


def test_scored_points_gives_each_stored_hit_its_own_truth_and_its_voxels_prediction():
    import numpy as np

    from pimm.datasets.transform import Compose
    from pimm.engines.metrics.panoptic import scored_points

    # 5 hits; hits 0 and 1 share a voxel but belong to different particles
    hits = dict(
        coord=np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [1.5, 0.1, 0.1],
                [2.5, 0.1, 0.1],
                [2.6, 0.2, 0.1],
            ]
        ),
        energy=np.ones((5, 1)),
        instance_particle=np.array([0, 1, 1, 2, 2]),
        segment_pid=np.array([3, 4, 4, 1, 1]),
    )
    keys = ("instance_particle", "segment_pid")
    batch = Compose(
        [
            dict(type="Copy", keys_dict={k: f"origin_{k}" for k in keys}),
            dict(
                type="GridSample",
                grid_size=1.0,
                mode="deterministic",
                return_inverse=True,
            ),
        ]
    )(hits)
    assert len(batch["coord"]) == 3 and len(batch["origin_instance_particle"]) == 5
    gt, gt_cls, voxel = scored_points(batch, *keys)
    assert gt.tolist() == [0, 1, 1, 2, 2] and gt_cls.tolist() == [3, 4, 4, 1, 1]
    pred = np.arange(10, 13)  # one prediction per voxel
    assert pred[voxel].tolist() == [pred[batch["inverse"][i]] for i in range(5)]
    assert (
        pred[voxel][0] == pred[voxel][1] and pred[voxel][3] == pred[voxel][4]
    )  # hits of a voxel share its prediction


def test_fm4npp_ari_is_sklearns_over_every_point_with_noise_and_unassigned_each_one_cluster():
    """FM4NPP's trainer: adjusted_rand_score(torch.unique(labels, return_inverse=True)[1], assignments) per event."""
    import numpy as np
    import torch
    from sklearn.metrics import adjusted_rand_score

    from pimm.engines.metrics.panoptic import event_row

    gt = np.array(
        [0, 0, 0, 1, 1, 1, -1, -1, 2, 2]
    )  # 2 hits unlabeled, 2 hits of the noise class (LAr LED)
    gt_cls = np.array([1, 1, 1, 2, 2, 2, 5, 5, 5, 5])
    pred = np.array([3, 3, 4, 4, 4, 4, -1, -1, -1, 3])
    pred_cls = np.where(pred >= 0, 1, -1)
    off = np.zeros(10, dtype=bool)
    row = event_row(gt, gt_cls, pred, pred_cls, off, off, 6, 5)
    truth = np.array(
        [0, 0, 0, 1, 1, 1, 9, 9, 9, 9]
    )  # unlabeled and LED hits: one true cluster
    expected = adjusted_rand_score(
        torch.unique(torch.as_tensor(truth), return_inverse=True)[1].numpy(), pred
    )
    assert row["ari"] == expected
    assert row["ari"] != adjusted_rand_score(
        truth, np.where(pred < 0, np.arange(10) + 100, pred)
    )  # not singletons


def test_noise_class_pq_on_a_hand_made_event():
    """Two tracks and noise; the noise is one segment of an extra class on both sides (sPHENIX: a noise head)."""
    import numpy as np
    import pytest

    from pimm.engines.metrics.panoptic import event_row, summarize_rows

    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    gt_cls = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    gt_noise = np.arange(13) >= 9  # track 2 is noise
    pred = np.array([5, 5, 5, 5, 7, 7, 7, 7, -1, 9, 9, 9, 7])
    pred_cls = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 1])
    pred_noise = pred == 9  # the noise head calls instance 9 noise
    row = event_row(gt, gt_cls, pred, pred_cls, gt_noise, pred_noise, 2, 2)
    # class 0: IoU 1; class 1: track 1 vs instance 7, IoU 4 / 6; noise: 3 of 4 hits, IoU 3 / 4
    assert row["pid_pq_truth"] == [1, 1, 1] and row["pid_pq_prediction"] == [1, 1, 1]
    assert row["pid_pq_true_positive"] == [1, 1, 1]
    assert row["pid_pq_iou_sum"] == pytest.approx([1.0, 4 / 6, 0.75])
    summary = summarize_rows([row], 2, 2)
    assert summary["pid_pq"] == pytest.approx((1 + 4 / 6 + 0.75) / 3)
    assert summary["headline"] == pytest.approx(
        2 * summary["ari"] * summary["pid_pq"] / (summary["ari"] + summary["pid_pq"])
    )
    # without the noise flag, track 2 is a thing: instance 9 (class 0) matches it at IoU 3 / 4 and there is no noise class
    row = event_row(
        gt, gt_cls, pred, pred_cls, np.zeros(13, bool), np.zeros(13, bool), 2, 2
    )
    assert row["pid_pq_truth"] == [2, 1, 0] and row["pid_pq_true_positive"] == [2, 1, 0]
