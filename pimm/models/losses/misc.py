"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union
from torch import Tensor
from .builder import LOSSES
from torch.nn.modules.loss import _WeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Standard multi-class cross-entropy over per-point class logits.

    Thin wrapper around :class:`torch.nn.CrossEntropyLoss`. ``forward(pred,
    target)`` expects ``pred`` of shape ``(N, C)`` (logits) and ``target`` of
    shape ``(N,)`` or ``(N, 1)`` (the trailing dim is squeezed); returns the
    reduced loss scaled by ``loss_weight``. Registered as ``CrossEntropyLoss`` --
    use in a ``criteria=[...]`` list (assembled by ``build_criteria``).

    Args:
        weight (list | None): Per-class rescaling weights, applied on the device of
            ``pred``. Defaults to ``None``.
        size_average (bool | None): Deprecated torch reduction flag. Defaults to
            ``None``.
        reduce (bool | None): Deprecated torch reduction flag. Defaults to
            ``None``.
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            Defaults to ``"mean"``.
        label_smoothing (float): Label-smoothing factor in ``[0, 1)``. Defaults
            to ``0.0``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        ignore_index (int): Target value excluded from the loss. Defaults to
            ``-1``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="CrossEntropyLoss", ignore_index=-1)])
            >>> pred = torch.randn(4, 3)             # (N=4 points, C=3 classes) logits
            >>> target = torch.tensor([0, 2, 1, -1]) # -1 is ignored
            >>> crit(pred, target)                   # scalar loss
            tensor(1.0995)
    """

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
        weight = torch.tensor(weight) if weight is not None else None
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
        # Criteria is not an nn.Module, so model.cuda() does not move the weight buffer
        return self.loss.to(pred.device)(pred, target.squeeze(-1)) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    """Label-smoothed cross-entropy with NaN-robust averaging.

    ``forward(pred, target)`` expects ``pred`` of shape ``(N, C)`` (logits) and
    ``target`` of shape ``(N,)`` (class indices); builds a smoothed one-hot
    target, computes the soft cross-entropy, and averages over finite entries.
    Registered as ``SmoothCELoss`` -- use in a ``criteria=[...]`` list.

    Args:
        smoothing_ratio (float): Smoothing mass spread over the non-target
            classes. Defaults to ``0.1``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="SmoothCELoss", smoothing_ratio=0.1)])
            >>> pred = torch.randn(4, 3)            # (N=4, C=3) logits
            >>> target = torch.tensor([0, 2, 1, 0]) # class indices (no ignore_index)
            >>> crit(pred, target)                  # scalar: smoothed soft-CE, averaged over finite entries
    """

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
    """Smooth-L1 (Huber) regression loss for continuous targets.

    ``forward(pred, target)`` returns ``loss_weight * smooth_l1_loss(pred,
    target)`` with the configured transition point and reduction; ``pred`` and
    ``target`` must broadcast. Registered as ``SmoothL1RegressionLoss`` -- use in
    a ``criteria=[...]`` list or as a per-head ``criterion`` in the instance
    regression losses.

    Args:
        beta (float): Transition point between the L2 and L1 regimes. Defaults to
            ``1.0``.
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            Defaults to ``"mean"``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="SmoothL1RegressionLoss", beta=1.0)])
            >>> pred = torch.zeros(4)
            >>> target = torch.ones(4)              # |err|=1 == beta -> 0.5 * err^2
            >>> crit(pred, target)                  # scalar loss
            tensor(0.5000)
    """

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
class GeodesicDirectionLoss(nn.Module):
    """Angular (geodesic) loss for a unit-direction regression head.

    ``forward(pred, target)`` expects ``pred``/``target`` of shape ``(M, D)``
    (typically ``D==3``), L2-normalizes both to unit vectors, and returns
    ``loss_weight * mean(arccos(clamp(cos, -c, c)))`` -- the true angular error in
    radians, where ``c = cos(clamp_deg)``. The cosine similarity is clamped so
    ``arccos``'s gradient (``1/sin angle``) stays finite; angles below
    ``clamp_deg`` incur approximately zero gradient. Computed in fp32 because the
    cosine is very close to one for aligned directions, where fp16 has too little
    resolution. Registered as ``GeodesicDirectionLoss``.

    Args:
        clamp_deg (float): Cosine-clamp angle in degrees. Defaults to ``0.1``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            ``"none"`` returns the per-instance angular error ``(M,)``. Defaults
            to ``"mean"``.
    """

    def __init__(self, clamp_deg=0.1, loss_weight=1.0, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        if not 0.0 < float(clamp_deg) < 90.0:
            raise ValueError("clamp_deg must be strictly between 0 and 90 degrees")
        self.clamp_deg = float(clamp_deg)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self._cos_clamp = float(math.cos(math.radians(self.clamp_deg)))

    def forward(self, pred, target):
        if pred.numel() == 0:
            return pred.float().sum() * 0.0
        p = F.normalize(pred.float(), dim=-1, eps=1e-6)
        t = F.normalize(target.float(), dim=-1, eps=1e-6)
        cos = (p * t).sum(dim=-1).clamp(-self._cos_clamp, self._cos_clamp)
        angular_error = self.loss_weight * torch.arccos(cos)
        if self.reduction == "none":
            return angular_error
        if self.reduction == "sum":
            return angular_error.sum()
        return angular_error.mean()


@LOSSES.register_module()
class L1RegressionLoss(nn.Module):
    """Mean-absolute-error (L1) regression loss for continuous targets.

    ``forward(pred, target)`` returns ``loss_weight * l1_loss(pred, target)``;
    ``pred`` and ``target`` must broadcast. Registered as ``L1RegressionLoss`` --
    use in a ``criteria=[...]`` list or as a per-head ``criterion``.

    Args:
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            Defaults to ``"mean"``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="L1RegressionLoss", loss_weight=1.0)])
            >>> crit(torch.zeros(4), torch.ones(4))  # mean |0 - 1| = 1.0
            tensor(1.)
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.l1_loss(pred, target, reduction=self.reduction)


@LOSSES.register_module()
class MSERegressionLoss(nn.Module):
    """Mean-squared-error (L2) regression loss for continuous targets.

    ``forward(pred, target)`` returns ``loss_weight * mse_loss(pred, target)``;
    ``pred`` and ``target`` must broadcast. Registered as ``MSERegressionLoss`` --
    use in a ``criteria=[...]`` list or as a per-head ``criterion``.

    Args:
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            Defaults to ``"mean"``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="MSERegressionLoss", loss_weight=1.0)])
            >>> crit(torch.zeros(4), torch.full((4,), 2.0))  # mean (0 - 2)^2 = 4.0
            tensor(4.)
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.mse_loss(pred, target, reduction=self.reduction)


@LOSSES.register_module()
class CrossEntropyHeadLoss(nn.Module):
    """Cross-entropy for an extra categorical per-query head.

    ``forward(pred, target)`` expects ``pred`` of shape ``(M, num_classes)``
    (matched-query logits) and ``target`` of shape ``(M,)`` (per-instance class
    indices, cast to long); returns ``loss_weight * cross_entropy(...)``. Intended
    as the per-head ``criterion`` for auxiliary categorical heads (beyond the
    primary PID head) in the unified instance loss. Registered as
    ``CrossEntropyHeadLoss``.

    Args:
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``).
            Defaults to ``"mean"``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        label_smoothing (float): Label-smoothing factor in ``[0, 1)``. Defaults
            to ``0.0``.
        ignore_index (int): Target value excluded from the loss. Defaults to
            ``-100``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="CrossEntropyHeadLoss")])
            >>> pred = torch.randn(3, 5)            # (M=3 matched queries, num_classes=5) logits
            >>> target = torch.tensor([0, 1, 2])    # per-instance class indices
            >>> crit(pred, target)                  # scalar loss
            tensor(2.1342)
            >>> # used as a per-query head criterion in the unified instance loss:
            >>> # query_heads=[dict(name="pid", kind="categorical",
            >>> #                    criterion=dict(type="CrossEntropyHeadLoss"))]
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
    """Binary focal loss for class-imbalanced binary targets.

    Focal loss (`Lin et al. 2017 <https://arxiv.org/abs/1708.02002>`_) for a
    single binary output. ``forward(pred, target)`` expects ``pred`` and
    ``target`` of matching shape ``(N,)`` (or broadcastable); ``pred`` is treated
    as logits when ``logits`` is ``True``, else as probabilities. Returns the
    (optionally mean-reduced) loss scaled by ``loss_weight``. Registered as
    ``BinaryFocalLoss`` -- use in a ``criteria=[...]`` list.

    Args:
        gamma (float): Focusing exponent that down-weights easy examples.
            Defaults to ``2.0``.
        alpha (float): Positive-class weight in ``(0, 1)``. Defaults to ``0.5``.
        logits (bool): Whether ``pred`` is raw logits (vs probabilities).
            Defaults to ``True``.
        reduce (bool): Mean-reduce the per-element loss when ``True``. Defaults to
            ``True``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        weight (torch.Tensor | None): Optional per-element BCE weight. Defaults to
            ``None``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="BinaryFocalLoss", gamma=2.0, alpha=0.5)])
            >>> pred = torch.zeros(4)               # logits (p=0.5 for all)
            >>> target = torch.tensor([0., 1., 0., 1.])
            >>> crit(pred, target)                  # scalar loss
            tensor(0.0866)
    """

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
            pred (torch.Tensor): The prediction with shape (N).
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is ``0 <= target[i] <= 1``;
                if containing class probabilities, same shape as the input.

        Returns:
            torch.Tensor: The calculated loss.
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
    """Multi-class focal loss over per-point class logits.

    Focal loss (`Lin et al. 2017 <https://arxiv.org/abs/1708.02002>`_) for
    multi-class classification. ``forward(pred, target)`` flattens ``pred`` to
    ``(N, C)`` (logits) and ``target`` to ``(N,)`` (class indices), applies the
    ``(1 - p_t) ** gamma`` focal modulation on top of cross-entropy, and reduces
    over non-ignored points. The optional class ``weight`` is indexed by the
    ground-truth label when ``penalize_pred`` is ``True``, otherwise by the
    predicted (argmax) label. Returns the loss scaled by ``loss_weight``.
    Registered as ``FocalLoss`` -- use in a ``criteria=[...]`` list.

    Args:
        weight (torch.Tensor | None): Per-class rescaling weights. Defaults to
            ``None``.
        size_average (bool | None): Deprecated torch reduction flag. Defaults to
            ``None``.
        reduce (bool | None): Deprecated torch reduction flag. Defaults to
            ``None``.
        reduction (str): Reduction mode (``"mean"``, ``"sum"``, ``"none"``); the
            ``"mean"`` path divides by the count of non-ignored targets. Defaults
            to ``"mean"``.
        gamma (float): Focusing exponent that down-weights easy examples.
            Defaults to ``2``.
        ignore_index (int): Target value excluded from the loss. Defaults to
            ``-1``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        penalize_pred (bool): Index the class ``weight`` by the ground-truth label
            instead of the predicted label. Defaults to ``False``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="FocalLoss", gamma=2.0, loss_weight=1.0)])
            >>> pred = torch.randn(4, 3)             # (N=4 points, C=3 classes) logits
            >>> target = torch.tensor([0, 2, 1, -1]) # -1 is ignored
            >>> crit(pred, target)                   # scalar loss
            tensor(0.9830)
    """

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
    """Soft multi-class Dice loss over per-point class probabilities.

    Dice loss (`V-Net, Milletari et al. 2016
    <https://arxiv.org/abs/1606.04797>`_). ``forward(pred, target)`` reshapes
    ``pred`` to ``(N, C)`` (logits, soft-maxed internally) and ``target`` to
    ``(N,)`` (class indices), drops ignored points, one-hot encodes the targets,
    and averages ``1 - Dice`` across classes. Returns the loss scaled by
    ``loss_weight``. Registered as ``DiceLoss`` -- use in a ``criteria=[...]``
    list, commonly alongside a cross-entropy or Lovasz term.

    Args:
        smooth (float): Smoothing constant added to the Dice numerator and
            denominator. Defaults to ``1``.
        exponent (float): Exponent applied to the per-class terms in the
            denominator. Defaults to ``2``.
        loss_weight (float): Global scale on the returned loss. Defaults to
            ``1.0``.
        ignore_index (int): Target value excluded from the loss. Defaults to
            ``-1``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.builder import build_criteria
            >>> crit = build_criteria([dict(type="DiceLoss", loss_weight=1.0)])
            >>> pred = torch.randn(4, 3)            # (N=4 points, C=3 classes) logits
            >>> target = torch.tensor([0, 2, 1, 0]) # class indices
            >>> crit(pred, target)                  # scalar 1 - Dice averaged over classes
            tensor(0.5031)
    """

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
