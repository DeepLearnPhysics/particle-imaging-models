from torch import nn

from pimm.models.default import DefaultSegmentorV2


def _frozen_segmentor():
    model = DefaultSegmentorV2.__new__(DefaultSegmentorV2)
    nn.Module.__init__(model)
    model.backbone = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(p=0.5))
    model.seg_head = nn.Linear(2, 2)
    model.criteria = nn.Identity()
    model.freeze_backbone = True
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def test_frozen_segmentor_keeps_backbone_in_eval_mode():
    model = _frozen_segmentor()

    model.train()

    assert model.training
    assert model.seg_head.training
    assert not model.backbone.training
