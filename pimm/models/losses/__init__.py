from .builder import build_criteria

from .misc import (
    BinaryFocalLoss,
    CrossEntropyHeadLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    L1RegressionLoss,
    MSERegressionLoss,
    SmoothCELoss,
    SmoothL1RegressionLoss,
)
from .lovasz import LovaszLoss
# InstanceSegmentationLoss is now a deprecated alias defined alongside the fast loss.
from .instance_fast import FastInstanceSegmentationLoss, InstanceSegmentationLoss
from .instance_regression_fast import FastInstanceSegmentationRegressionLoss
from .instance_unified_fast import FastUnifiedInstanceLoss
