"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union
from torch import Tensor
from .builder import LOSSES
from torch.nn.modules.loss import _WeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target.squeeze(-1)) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class SmoothL1RegressionLoss(nn.Module):
    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.smooth_l1_loss(
            pred, target, beta=self.beta, reduction=self.reduction
        )


@LOSSES.register_module()
class L1RegressionLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.l1_loss(pred, target, reduction=self.reduction)


@LOSSES.register_module()
class MSERegressionLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.mse_loss(pred, target, reduction=self.reduction)


@LOSSES.register_module()
class CrossEntropyHeadLoss(nn.Module):
    """Cross-entropy for an extra categorical per-query head.

    Expects ``pred`` of shape ``(M, num_classes)`` (matched-query logits) and
    ``target`` of shape ``(M,)`` (per-instance class indices). Use for
    auxiliary categorical heads beyond the primary PID head.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, label_smoothing=0.0,
                 ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.loss_weight * F.cross_entropy(
            pred,
            target.long(),
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0, weight=None):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight
        self.weight = weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none", weight=self.weight)
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none", weight=self.weight)
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        gamma: float = 2,
        ignore_index: int = -1,
        loss_weight: float = 1.0,
        penalize_pred: bool = False,
    ):
        super().__init__(
            torch.tensor(weight) if weight is not None else None,
            size_average,
            reduce,
            reduction,
        )
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.penalize_pred = penalize_pred

    def forward(self, pred, target):
        if self.weight is not None and self.weight.device != pred.device:
            self.weight = self.weight.to(pred.device)
        flattened_logits = pred.reshape(-1, pred.shape[-1])
        flattened_labels = target.view(-1).long()

        p_t = flattened_logits.softmax(dim=-1)
        ce_loss = F.cross_entropy(
            flattened_logits,
            flattened_labels,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # -log(p_t)

        if self.weight is not None:
            alpha_t = self.weight[
                flattened_labels
                if self.penalize_pred
                else flattened_logits.argmax(dim=-1)
            ]
        else:
            alpha_t = 1.0

        loss = (
            alpha_t
            * ((1 - p_t[torch.arange(p_t.shape[0]), flattened_labels]) ** self.gamma)
            * ce_loss
        )

        if self.reduction == "mean":
            loss = loss.sum() / target.ne(self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
