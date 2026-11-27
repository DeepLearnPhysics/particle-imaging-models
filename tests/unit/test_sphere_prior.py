"""Fixed training-prior scenes and retained synthetic evaluation metrics."""

import pytest
import torch

from pimm.datasets import collate_fn
from pimm.datasets.sphere_prior import SpherePriorDataset
from pimm.datasets.sphere_prior.geometry import cap_oracle_iou, to_sphere
from pimm.engines.hooks.eval.synthetic import (
    OperatingPoint,
    score_event,
    split_truth,
    summarize,
)


@pytest.fixture(scope="module")
def dataset():
    return SpherePriorDataset(length=64, split="val")


def test_scenes_are_deterministic(dataset):
    first, second = dataset[5], dataset[5]
    assert torch.equal(first["feat"], second["feat"])
    assert torch.equal(first["inst_pe"], second["inst_pe"])


def test_seeds_and_splits_give_independent_streams():
    a = SpherePriorDataset(length=4, seed=0)
    b = SpherePriorDataset(length=4, seed=1)
    c = SpherePriorDataset(length=4, seed=0, split="val")
    assert not torch.equal(a[0]["feat"][:8], b[0]["feat"][:8])
    assert not torch.equal(a[0]["feat"][:8], c[0]["feat"][:8])


def test_features_lie_on_the_layernorm_sphere(dataset):
    for index in range(16):
        feat = dataset[index]["feat"]
        assert feat.shape == (2048, 256)
        assert torch.allclose(feat.norm(dim=1), torch.ones(len(feat)), atol=1e-5)
        assert feat.sum(dim=1).abs().max() < 1e-4


def test_groups_are_exclusive_equal_and_within_bounds(dataset):
    for index in range(32):
        sample = dataset[index]
        owner, inst_pe = sample["instance"], sample["inst_pe"]
        num_groups = int(sample["ring_valid"].sum())
        assert 2 <= num_groups <= 40
        sizes = torch.bincount(owner[owner >= 0], minlength=num_groups)
        assert len(sizes) == num_groups and sizes.max() - sizes.min() <= 1
        # one-hot ownership; background rows and unused slots are empty
        assert torch.equal(
            inst_pe[owner >= 0].sum(dim=1), torch.ones(int((owner >= 0).sum()))
        )
        assert (inst_pe[owner < 0] == 0).all() and (inst_pe[:, num_groups:] == 0).all()
        assert (sample["inst_class"] == 0).all()


def test_cap_oracle_isolates_a_tight_cluster():
    rng = torch.Generator().manual_seed(0)
    centre = torch.randn(32, generator=rng)
    inside = centre + 0.1 * torch.randn((200, 32), generator=rng)
    outside = torch.randn((200, 32), generator=rng)
    sphere = to_sphere(torch.cat((inside, outside)))
    members = torch.arange(400) < 200
    assert cap_oracle_iou(sphere, members) > 0.95


def _row(
    matched: int, true: int = 4, kept: int = 4, mask_recalled: int | None = None
) -> dict:
    return dict(
        name="e",
        num_true=true,
        num_kept=kept,
        num_matched=matched,
        num_mask_recalled=matched if mask_recalled is None else mask_recalled,
        matched_iou_sum=0.8 * matched,
        num_oracle_defined=true,
        num_oracle_isolated=true,
        cap_threshold=[0.1],
        cap_sharpness=[50.0],
        ari=0.0,
    )


def test_no_matches_score_zero_f1_not_nan():
    assert summarize([_row(matched=0)])["f1"] == 0.0


def test_mask_recall_ignores_confidence():
    metrics = summarize([_row(matched=0, mask_recalled=3)])
    assert metrics["all/mask_recall"] == pytest.approx(0.75)
    assert metrics["all/recall"] == 0.0


def test_summary_reports_instance_count_bins():
    metrics = summarize([_row(matched=2, true=4), _row(matched=1, true=25, kept=5)])
    assert (
        "instances_3-5/mask_recall" in metrics
        and "instances_21+/mask_recall" in metrics
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_splits_use_the_same_prior_and_collation(split):
    dataset = SpherePriorDataset(length=2, split=split, seed=19)
    samples = [dataset[index] for index in range(2)]
    batch = collate_fn(samples)
    assert batch["feat"].shape == (4096, 256)
    assert batch["inst_pe"].shape == (4096, 48)
    assert batch["offset"].tolist() == [2048, 4096]
    truth = split_truth(batch)
    rows = []
    for sample, event in zip(samples, truth):
        assert 2 <= len(event.masks) <= 40
        assert torch.equal(event.owner, sample["instance"])
        prediction = dict(
            mask_logits=event.masks.float() * 2 - 1,
            class_probs=torch.tensor([[1.0, 0.0]]).repeat(len(event.masks), 1),
            cap_threshold=torch.zeros(len(event.masks)),
            cap_sharpness=torch.ones(len(event.masks)),
        )
        rows.append(
            score_event(prediction, event, OperatingPoint(), [1.0] * len(event.masks))
        )
    metrics = summarize(rows)
    assert metrics["events"] == 2
    assert metrics["f1"] == 1.0
    assert metrics["ari"] == 1.0
    assert metrics["all/mask_recall"] == 1.0
    assert metrics["all/oracle_recall"] == 1.0


def test_index_bounds_and_split_validation():
    with pytest.raises(ValueError, match="split"):
        SpherePriorDataset(length=1, split="other")
    data = SpherePriorDataset(length=1)
    for index in (-1, 1):
        with pytest.raises(IndexError):
            data[index]
