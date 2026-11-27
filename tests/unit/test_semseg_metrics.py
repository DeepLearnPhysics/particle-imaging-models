from pathlib import Path

import numpy as np
import pytest

from pimm.engines.metrics import compute_semseg_metrics
from pimm.utils.config import Config

ROOT = Path(__file__).resolve().parents[2]


def test_ignored_points_are_not_scored():
    pred = np.array([0, 1, 1, 1])
    target = np.array([0, 1, -1, -1])
    metrics = compute_semseg_metrics(pred, target, 2, ignore_index=-1)
    np.testing.assert_allclose(metrics.iou_class, [1.0, 1.0])
    np.testing.assert_allclose(metrics.precision_class, [1.0, 1.0])
    np.testing.assert_allclose(metrics.f1_class, [1.0, 1.0])
    assert metrics.all_acc == pytest.approx(1.0)


@pytest.mark.parametrize("detector", ["lartpc", "sphenix", "wc", "next"])
def test_pretraining_batches_split_over_four_gpus(detector):
    cfg = Config.fromfile(str(ROOT / "configs" / "panda" / "v2" / "pretrain" / f"{detector}.py"))
    for key in ("batch_size", "batch_size_val"):
        value = cfg.get(key)
        assert value is None or value % 4 == 0, f"{key}={value}"
