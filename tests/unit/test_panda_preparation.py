import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pimm.datasets import collate_fn
from pimm.datasets.transform.preparation import SonataPreparation
from pimm.models.panda_v2.panda_ssl_v2m1 import PandaV2SelfDistillation

MODEL = dict(
    backbone=dict(type="panda-encoder-v2m1"),
    observable=True,
    up_cast_level=3,
    mask_size=0.4,
    mask_ratio=0.0,
    mask_jitter=0.0,
    match_max_r=0.2,
)


def sample():
    coord = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]).repeat(2, 1)
    local = coord[:1].repeat(6, 1)
    return dict(
        global_coord=coord,
        global_origin_coord=coord.clone(),
        global_feat=torch.cat([coord, torch.ones(4, 1)], 1),
        global_offset=torch.tensor([2, 4]),
        local_coord=local,
        local_origin_coord=local.clone(),
        local_feat=torch.cat([local, torch.ones(6, 1)], 1),
        local_offset=torch.arange(1, 7),
        grid_size=torch.tensor([0.1]),
    )


def prepare(samples, **settings):
    transform = SonataPreparation(dict(MODEL, **settings))
    return collate_fn([transform(copy.deepcopy(value)) for value in samples])


def test_preparation_collates_views_and_preserves_the_batch():
    samples = [sample(), sample()]
    expected = collate_fn(copy.deepcopy(samples))
    actual = prepare(samples)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    plan = actual["sonata_plan"]
    assert torch.equal(plan["visible"], torch.arange(8))
    assert plan["settings"]["num_global_view"] == 2
    assert plan["settings"]["num_local_view"] == 6
    teacher = plan["sites"]["teacher"]
    assert torch.equal(teacher["row_offset"], torch.tensor([2, 4, 6, 8]))
    assert torch.equal(teacher["offset"], torch.tensor([2, 4, 6, 8]))
    assert torch.equal(teacher["grid_coord"][:2], torch.tensor([[0, 0, 0], [1, 0, 0]]))
    assert plan["mask_match"].shape == (8, 2)
    assert plan["local_match"].shape == (12, 2)


def test_matches_use_the_pooled_sites_below_the_input_level():
    plan = prepare([sample()], up_cast_level=2)["sonata_plan"]
    # the two sites of each global view share one second-stage voxel
    assert torch.equal(plan["mask_match"], torch.tensor([[0, 0], [1, 1]]))
    assert torch.equal(plan["roll_mask_match"], torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(plan["local_match"][:, 1], torch.zeros(6, dtype=torch.long))


def test_a_fully_masked_view_keeps_its_highest_priority_patch():
    transform = SonataPreparation(dict(MODEL, mask_size=0.05, mask_ratio=1.0))
    value = transform(sample())
    scores = value["sonata_plan"].scores
    plan = collate_fn([value])["sonata_plan"]
    # two one-point patches per view; the visible row is the patch with the higher score
    expected = [view * 2 + int(np.argmax(scores[view * 2 : view * 2 + 2])) for view in range(2)]
    assert plan["visible"].tolist() == expected


def test_empty_views_are_rejected():
    value = sample()
    value["global_offset"] = torch.tensor([0, 4])
    with pytest.raises(ValueError, match="every global and local view"):
        SonataPreparation(MODEL)(value)


def test_unobservable_masks_share_the_teacher_geometry_without_jitter():
    plan = prepare([sample()], observable=False, mask_size=0.05, mask_ratio=0.5)["sonata_plan"]
    assert plan["mask"].any() and not plan["mask"].view(2, 2).all(1).any()
    assert plan["sites"]["student"] is plan["sites"]["teacher"]
    assert torch.equal(plan["coord"], sample()["global_coord"])
    assert torch.equal(plan["mask_match"], torch.arange(4)[:, None].repeat(1, 2))


def test_unobservable_jitter_moves_only_masked_rows():
    np.random.seed(3)
    plan = prepare(
        [sample()], observable=False, mask_size=0.05, mask_ratio=0.5, mask_jitter=0.01
    )["sonata_plan"]
    shift = plan["coord"] - sample()["global_coord"]
    assert torch.equal(shift.ne(0).any(1), plan["mask"])
    assert float(shift.max()) <= 0.02
    assert torch.equal(plan["mask_match"], torch.arange(4)[:, None].repeat(1, 2))


def test_custom_collation_waits_for_fields_and_skips_private_metadata():
    class Plan:
        @staticmethod
        def collate(values, batch):
            return batch["x"] + len(values)

    result = collate_fn(
        [
            dict(x=torch.tensor([1]), plan=Plan(), _private=Plan()),
            dict(x=torch.tensor([2]), plan=Plan(), _private=Plan()),
        ]
    )
    assert torch.equal(result["plan"], torch.tensor([3, 4]))
    assert "_private" not in result


def test_settings_come_from_the_model_config_with_its_defaults():
    settings = SonataPreparation(dict(backbone=dict(type="panda-encoder-v2m1"))).settings
    assert settings == dict(
        strides=(2, 1, 4),
        observable=False,
        up_cast_level=2,
        mask_size=0.4,
        mask_ratio=0.7,
        mask_jitter=0.01,
        match_max_r=0.08,
        observable_match_max_r=0.08,
    )


def test_prepared_settings_must_match_the_model():
    plan = prepare([sample()])["sonata_plan"]
    backbone = SimpleNamespace(strides=(2, 1, 4))
    model = SimpleNamespace(
        **{key: value for key, value in plan["settings"].items() if key != "strides"},
        student=SimpleNamespace(backbone=backbone),
        teacher=SimpleNamespace(backbone=backbone),
    )
    PandaV2SelfDistillation.check_plan(model, plan["settings"])
    model.mask_ratio = 0.5
    with pytest.raises(ValueError, match="mask_ratio"):
        PandaV2SelfDistillation.check_plan(model, plan["settings"])
    model.mask_ratio, backbone.strides = 0.0, (2, 2, 2)
    with pytest.raises(ValueError, match="strides"):
        PandaV2SelfDistillation.check_plan(model, plan["settings"])
