"""PointClassificationEvaluator's pooled metrics equal FM4NPP's (sklearn per-class precision / recall, their means,
accuracy) and sklearn's macro F1, and DefaultSegmentorV2 gives its evaluation output at the original points."""

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pimm.engines.hooks.eval.semantic import pooled_metrics


def test_pooled_metrics_match_sklearn_including_degenerate_predictions():
    rng = np.random.default_rng(0)
    for trial in range(200):
        k, n = rng.integers(2, 6), rng.integers(1, 400)
        true = rng.integers(0, k, n)
        pred = (
            np.zeros(n, int)
            if trial % 7 == 0
            else np.where(rng.random(n) < 0.6, true, rng.integers(0, k, n))
        )
        confusion = np.zeros((k, k), int)
        np.add.at(confusion, (true, pred), 1)
        m = pooled_metrics(confusion)
        assert np.isclose(
            m["macro_precision"],
            precision_score(true, pred, average=None, zero_division=0).mean(),
        )
        assert np.isclose(
            m["macro_recall"],
            recall_score(true, pred, average=None, zero_division=0).mean(),
        )
        assert np.isclose(m["accuracy"], accuracy_score(true, pred))
        assert np.isclose(
            m["macro_f1"], f1_score(true, pred, average="macro", zero_division=0)
        )


def _segmentor(heads=None):
    from torch import nn

    from pimm.models.default import DefaultSegmentorV2

    model = DefaultSegmentorV2.__new__(DefaultSegmentorV2)
    nn.Module.__init__(model)
    model.backbone, model.freeze_backbone = nn.Identity(), False  # logits = the voxel features
    model.heads = dict(heads or {})
    loss = lambda logits, target: logits.sum() * 0
    if heads:
        model.seg_head = nn.ModuleDict({key: nn.Identity() for key in heads})
        model.head_criteria = {key: loss for key in heads}
        model.head_labels = dict(heads)
    else:
        model.seg_head, model.criteria = nn.Identity(), loss
    return model.eval()


def _batch(label):
    # event 0 has 2 voxels and 3 points, event 1 has 3 voxels and 3 points; inverse is local to each event
    return {
        "feat": torch.tensor([[10.0], [11.0], [20.0], [21.0], [22.0]]),
        "offset": torch.tensor([2, 5]),
        "origin_offset": torch.tensor([3, 6]),
        "inverse": torch.tensor([1, 0, 1, 2, 0, 2]),
        label: torch.tensor([0, 1, 0, 1, 2]),
        f"origin_{label}": torch.tensor([1, 0, 1, 2, 0, 2]),
    }


def test_segmentor_evaluates_each_original_point_with_its_voxel():
    batch = _batch("segment")
    out = _segmentor()(batch)
    assert out["seg_logits"].view(-1).tolist() == [11, 10, 11, 22, 20, 22]
    assert out["segment"] is batch["origin_segment"] and out["offset"] is batch["origin_offset"]
    out = _segmentor(dict(mlp="segment_pid"))(_batch("segment_pid"))
    assert out["seg_logits"]["mlp"].view(-1).tolist() == [11, 10, 11, 22, 20, 22]
    assert out["segment_pid"].tolist() == [1, 0, 1, 2, 0, 2]
    batch.pop("inverse")
    with pytest.raises(KeyError, match=r"GridSample\(return_inverse=True\)"):
        _segmentor()(batch)
