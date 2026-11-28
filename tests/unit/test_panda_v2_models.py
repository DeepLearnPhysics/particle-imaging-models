import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pimm.models.builder import MODELS
from pimm.models.losses.instance import FastUnifiedInstanceLoss
from pimm.models.panda_detector.panda_detector_v1m1 import PandaV1Detector
from pimm.models.utils.structure import Point


@pytest.fixture
def torch_attention():
    from pimm.ops import set_attention_backend

    set_attention_backend("torch")
    yield
    set_attention_backend("flash")


def test_panda_v2_registry_preserves_public_detector():
    assert MODELS.get("panda-detector-v1m1") is PandaV1Detector
    assert all(
        MODELS.get(name) is None
        for name in ("detector-v4", "detector-v5", "detector-v5m2")
    )
    for name in (
        "panda-encoder-v2m1",
        "panda-detector-v2m1",
        "panda-ssl-v2m1",
    ):
        assert MODELS.get(name) is not None
    for name in (
        "LitePT-v2m1",
        "Sonata-v1m5",
        "panda-sphere-prior-pretrain",
        "PandaV2Detector",
        "panda-synthetic-v2m1",
        "detector-v3m2",
    ):
        assert MODELS.get(name) is None


@pytest.mark.gpu
def test_encoder_size_arguments_control_multiscale_features(torch_attention):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from pimm.models.panda_v2 import PandaV2Encoder

    model = (
        PandaV2Encoder(
            channels=(12, 24, 48, 72),
            depths=(1, 2, 2, 1),
            heads=(1, 2, 4, 6),
        )
        .cuda()
        .eval()
    )
    grid = torch.cartesian_prod(*(torch.arange(4, device="cuda"),) * 3)
    features = torch.randn(len(grid), 4, device="cuda", requires_grad=True)
    point = model(
        dict(
            feat=features,
            coord=grid.float(),
            origin_coord=grid.float() + 1,
            grid_coord=grid,
            offset=torch.tensor([len(grid)], device="cuda"),
        )
    )
    # the coarsest stage: one voxel, with the mean coordinates of its children (equal-sized here)
    assert point.feat.shape == (1, 72) and torch.equal(point.offset.cpu(), torch.tensor([1]))
    assert torch.equal(point.grid_coord, torch.zeros(1, 3, dtype=torch.long, device="cuda"))
    torch.testing.assert_close(point.coord, grid.float().mean(0, keepdim=True))
    torch.testing.assert_close(point.origin_coord, grid.float().mean(0, keepdim=True) + 1)
    result = point.upcast().feat
    assert result.shape == (len(grid), 156)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0


def tiny_samples(count=2, views=(2, 2), points=(300, 120)):
    generator = torch.Generator().manual_seed(11)
    samples = []
    for _ in range(count):
        sample = dict(grid_size=torch.tensor([0.01]))
        for prefix, n, size in zip(("global", "local"), views, points):
            coord = torch.rand(n * size, 3, generator=generator) * (0.6 if prefix == "global" else 0.3)
            sample.update(
                {
                    f"{prefix}_coord": coord,
                    f"{prefix}_origin_coord": coord.clone(),
                    f"{prefix}_feat": torch.cat([coord, torch.rand(n * size, 1, generator=generator)], 1),
                    f"{prefix}_offset": torch.arange(1, n + 1) * size,
                }
            )
        samples.append(sample)
    return samples


@pytest.mark.gpu
@pytest.mark.parametrize(
    "conv_stages, strides, up_cast_level, observable, mask_jitter",
    [
        (2, (2, 1, 4), 3, True, 0.0),
        (1, (2, 1, 4), 2, True, 0.0),
        (3, (2, 2, 2), 2, True, 0.0),
        (2, (2, 2, 2), 3, False, 0.01),
        (2, (2, 1, 4), 2, False, 0.01),
        (2, (2, 1, 4), 1, False, 0.0),
    ],
)
def test_self_distillation_trains_on_prepared_batches(
    conv_stages, strides, up_cast_level, observable, mask_jitter, torch_attention
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from types import SimpleNamespace
    from pimm.datasets import collate_fn
    from pimm.datasets.transform.preparation import SonataPreparation, _view_sites
    from pimm.distributed import move_batch_to_device
    from pimm.models.panda_v2 import PandaV2SelfDistillation

    channels = (12, 24, 48, 72)
    config = dict(
        backbone=dict(
            type="panda-encoder-v2m1",
            grid_size=0.01,
            channels=channels,
            depths=(1, 2, 2, 1),
            heads=(1, 2, 4, 6),
            strides=strides,
            conv_stages=conv_stages,
            mask_token=not observable,
        ),
        head_in_channels=sum(channels[3 - up_cast_level :]),
        head_hidden_channels=32,
        head_embed_channels=16,
        head_num_prototypes=32,
        num_local_view=2,
        observable=observable,
        up_cast_level=up_cast_level,
        mask_size=0.1,
        mask_ratio=0.6,
        mask_jitter=mask_jitter,
        match_max_r=0.05,
    )
    transform = SonataPreparation(copy.deepcopy(config))
    torch.manual_seed(0)
    model = PandaV2SelfDistillation(**copy.deepcopy(config)).cuda()
    model.trainer = SimpleNamespace(
        cfg=SimpleNamespace(scheduler=SimpleNamespace(total_steps=100)),
        global_step=0,
        start_epoch=0,
        train_loader=[],
        writer=None,
    )
    model.before_train()
    model.before_step()
    np.random.seed(0)
    batch = collate_fn([transform(sample) for sample in tiny_samples()])
    cpu_batch = batch
    batch = move_batch_to_device(batch, torch.device("cuda"))
    output = model(batch)
    assert torch.isfinite(output["loss"]) and len(batch["sonata_plan"]["mask_match"]) > 0
    output["loss"].backward()
    grads = [p.grad for p in model.student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    kinds = [type(stage[0]).__name__ for stage in model.student.backbone.blocks]
    assert kinds == ["ConvBlock"] * conv_stages + ["AttnBlock"] * (4 - conv_stages)

    # the encoder's loss level has the preparation's views and origins, in the same order
    plan = cpu_batch["sonata_plan"]
    _, (origin, ends) = _view_sites(
        cpu_batch["global_coord"],
        cpu_batch["global_origin_coord"],
        cpu_batch["global_offset"],
        0.01,
        plan["settings"],
    )
    teacher = Point(
        feat=batch["global_feat"],
        coord=batch["global_coord"],
        origin_coord=batch["global_origin_coord"],
        offset=batch["global_offset"],
        batch=batch["sonata_plan"]["sites"]["teacher"]["row_batch"],
        grid_size=batch["grid_size"][0],
    )
    with torch.no_grad():
        level = model.encode_view(model.teacher, teacher, {}, batch["sonata_plan"], "teacher")
    assert torch.equal(level.offset.cpu(), torch.from_numpy(ends))
    torch.testing.assert_close(level.origin_coord.double().cpu(), torch.from_numpy(origin), atol=1e-6, rtol=0)


def test_public_instance_classification_remains_the_default():
    masks = torch.tensor([[4.0, 4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, 4.0], [-4.0] * 4])
    logits = torch.tensor([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.0, 2.0]])
    pred = {"pred_masks": [masks], "pred_logits": [logits]}
    truth = {
        "instance": torch.tensor([0, 0, 1, 1]),
        "segment": torch.tensor([0, 0, 1, 1]),
    }
    criterion = FastUnifiedInstanceLoss(
        query_heads=[], loss_weight_focal=0, loss_weight_dice=0
    )
    actual, _ = criterion(pred, truth)
    expected = (
        2 * F.cross_entropy(logits[:2], torch.tensor([0, 1]))
        + 0.1 * F.cross_entropy(logits[2:], torch.tensor([2]))
    ) / 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    modern, _ = FastUnifiedInstanceLoss(
        query_heads=[], panda_v2=True, loss_weight_focal=0, loss_weight_dice=0
    )(pred, truth)
    assert not torch.isclose(actual, modern)


def test_consistent_matching_reuses_final_assignment_for_auxiliary_masks():
    masks = torch.tensor([[4.0, 4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, 4.0]])
    pred = {"pred_masks": [masks], "aux_outputs": [{"pred_masks": [masks.flip(0)]}]}
    truth = {"instance": torch.tensor([0, 0, 1, 1])}
    independent, _ = FastUnifiedInstanceLoss(query_heads=[])(pred, truth)
    consistent, _ = FastUnifiedInstanceLoss(query_heads=[], consistent_matching=True)(
        pred, truth
    )
    assert consistent > independent + 1


def test_panda_v2_empty_truth_still_supervises_no_object():
    masks = torch.randn(2, 4, requires_grad=True)
    logits = torch.randn(2, 3, requires_grad=True)
    pred = {"pred_masks": [masks], "pred_logits": [logits]}
    truth = {"instance": torch.full((4,), -1), "segment": torch.full((4,), -1)}
    loss, _ = FastUnifiedInstanceLoss(query_heads=[], panda_v2=True)(pred, truth)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert masks.grad is not None and torch.equal(masks.grad, torch.zeros_like(masks))
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_varlen_attention_fallback_isolates_events_and_gradients():
    from pimm.ops import varlen_attention

    torch.manual_seed(7)
    q = torch.randn(3, 2, 4, requires_grad=True)
    k = torch.randn(5, 2, 4, requires_grad=True)
    v = torch.randn(5, 2, 4, requires_grad=True)
    bounds_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    bounds_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    output = varlen_attention(q, k, v, bounds_q, bounds_k, 2, 3, fused=False)
    expected = F.scaled_dot_product_attention(
        q[:2].transpose(0, 1), k[:3].transpose(0, 1), v[:3].transpose(0, 1)
    ).transpose(0, 1)
    torch.testing.assert_close(output[:2], expected)
    output[:2].sum().backward()
    assert torch.equal(k.grad[3:], torch.zeros_like(k.grad[3:]))
    assert torch.equal(v.grad[3:], torch.zeros_like(v.grad[3:]))


def test_panoptic_decoding_accepts_half_precision_regression():
    from pimm.models.panda_v2.detector_utils.postprocess import postprocess_panoptic

    masks = torch.tensor([[4.0, 4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, 4.0]])
    classes = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    regression = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    output = postprocess_panoptic(
        masks, classes, 4, pred_regression={"vector": regression}, min_points=1
    )
    expected = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    torch.testing.assert_close(output["instance_vector"], expected, rtol=0, atol=0)


def test_detector_injects_specs_into_consistent_matching_loss():
    from pimm.models.panda_v2.panda_detector_v2m1 import PandaV2Detector

    model = PandaV2Detector(
        full_in_channels=4,
        hidden_channels=8,
        num_heads=2,
        num_queries=2,
        num_classes=2,
        overlap=True,
        backbone={"type": torch.nn.Identity},
        criteria=[{"type": "FastUnifiedInstanceLoss", "consistent_matching": True}],
    )
    loss = model.criteria_by_label["particle"].criteria[0]
    assert (
        loss.consistent_matching and loss.criterion.overlap and loss.criterion.panda_v2
    )


def test_detector_without_backbone_trains_and_predicts_spherical_caps(torch_attention):
    from pimm.models.panda_v2.panda_detector_v2m1 import PandaV2Detector

    model = PandaV2Detector(
        full_in_channels=8,
        hidden_channels=8,
        num_heads=2,
        labels=("instance",),
        num_queries=3,
        num_classes=1,
        label_configs=dict(
            instance=dict(
                instance_key="instance", segment_key="inst_class", overlap=True
            )
        ),
        pos_emb=False,
        qkv_bias=True,
        criteria=[
            dict(
                type="FastUnifiedInstanceLoss",
                query_heads=[],
                membership_key="inst_pe",
                valid_key="ring_valid",
                segment_key="inst_class",
            )
        ],
    )
    owners = torch.tensor([0, 0, 0, 1, 1, 1])
    batch = dict(
        feat=torch.randn(6, 8),
        coord=torch.zeros(6, 3),
        offset=torch.tensor([6]),
        instance=owners,
        inst_pe=F.one_hot(owners, 2).float(),
        ring_valid=torch.ones(2, dtype=torch.bool),
        inst_class=torch.zeros(2, dtype=torch.long),
    )
    assert isinstance(model.backbone, torch.nn.Identity)
    output = model(batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.decoder.full_point_proj.weight.grad.abs().sum() > 0
    prediction = model.eval().predict(batch)[0]
    assert prediction["mask_logits"].shape == (3, 6)
    assert prediction["class_probs"].shape == (3, 2)
    assert prediction["sphere_points"].shape == (6, 8)
    for key in ("cap_axis", "cap_threshold", "cap_sharpness"):
        assert torch.isfinite(prediction[key]).all()


def test_checkpoint_loader_additional_source_preserves_other_parameters(tmp_path):
    import logging
    from types import SimpleNamespace
    from pimm.engines.hooks.checkpoint import CheckpointLoader

    model = torch.nn.ModuleDict(
        dict(backbone=torch.nn.Linear(2, 2), decoder=torch.nn.Linear(2, 2))
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    source = tmp_path / "decoder.pth"
    torch.save(
        {"state_dict": {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}}, source
    )
    hook = CheckpointLoader(
        source=str(source),
        replacements={"weight": "decoder.weight", "bias": "decoder.bias"},
    )
    hook.trainer = SimpleNamespace(
        model=model,
        cfg=SimpleNamespace(resume=False),
        logger=logging.getLogger(__name__),
    )
    hook.before_train()
    torch.testing.assert_close(model.decoder.weight, torch.ones(2, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        model.backbone.weight, before["backbone.weight"], rtol=0, atol=0
    )
    hook.trainer.cfg.resume = True
    model.decoder.weight.data.zero_()
    hook.before_train()
    assert not model.decoder.weight.any()


def test_detector_autocast_preserves_float32_mask_logits(torch_attention):
    from pimm.models.panda_v2.panda_detector_v2m1 import PandaV2Detector

    model = PandaV2Detector(
        full_in_channels=4,
        hidden_channels=8,
        num_heads=2,
        num_queries=3,
        num_classes=2,
        qkv_bias=True,
    ).eval()
    batch = dict(
        feat=torch.randn(6, 4), coord=torch.zeros(6, 3), offset=torch.tensor([6])
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
    assert output["pred_masks"][0].dtype == torch.float32
    assert output["seg_logits"].dtype == torch.float32


@pytest.mark.gpu
def test_torch_attention_backend_matches_flash_without_calling_it(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip(
        "flash_attn_turing" if torch.cuda.get_device_capability() == (7, 5) else "flash_attn"
    )
    import pimm.ops.attention as attention
    import pimm.ops.triton.window_attention as window
    from pimm.models.panda_v2.panda_detector_v2m1 import PandaV2Detector
    from pimm.ops import set_attention_backend

    torch.manual_seed(0)
    model = (
        PandaV2Detector(
            full_in_channels=156,
            hidden_channels=32,
            num_heads=2,
            num_queries=4,
            num_classes=2,
            qkv_bias=True,
            backbone=dict(
                type="panda-encoder-v2m1",
                channels=(12, 24, 48, 72),
                depths=(1, 2, 2, 1),
                heads=(1, 2, 4, 6),
            ),
        )
        .cuda()
        .eval()
    )
    counts = (1500, 300)
    grid = torch.cat([torch.randperm(32**3)[:n] for n in counts])
    grid = torch.stack([grid // 32**2, grid // 32 % 32, grid % 32], 1).cuda()
    batch = dict(
        feat=torch.randn(len(grid), 4, device="cuda"),
        coord=grid.float(),
        grid_coord=grid,
        offset=torch.tensor(counts, device="cuda").cumsum(0),
    )

    def forward():
        torch.manual_seed(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(dict(batch), return_point=True)
        return out["point"].feat, out["pred_masks"][0], out["pred_logits"][0]

    flash = forward()

    def no_flash(*_):
        raise AssertionError("FlashAttention ran with the torch backend")

    monkeypatch.setattr(attention, "_flash", no_flash)
    monkeypatch.setattr(window, "_flash", no_flash)
    set_attention_backend("torch")
    try:
        reference = forward()
    finally:
        set_attention_backend("flash")
    for actual, expected in zip(reference, flash):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=1e-3)
