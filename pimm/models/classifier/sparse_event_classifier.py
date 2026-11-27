"""Attach an event loss to a sparse-convolution backbone's per-event logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from pimm.models.losses import build_criteria
from pimm.models.builder import MODELS, build_model


@MODELS.register_module("SparseEventClassifier")
class SparseEventClassifier(nn.Module):
    """Wrap a sparse-conv backbone (returns per-event logits) with a loss."""

    def __init__(self, backbone, criteria=None, label_key="event_label"):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.label_key = label_key

    def forward(self, input_dict, return_point=False):
        logits = self.backbone(input_dict)  # (n_events, num_classes)
        result = dict()
        if self.label_key in input_dict.keys():
            labels = input_dict[self.label_key].detach().long().view(-1)
            result["loss"] = self.criteria(logits, labels)
            if not self.training:
                result["cls_logits"] = logits
        else:
            result["cls_logits"] = logits
        return result
