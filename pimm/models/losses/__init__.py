from .builder import build_criteria

from .misc import (
    BinaryFocalLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    L1RegressionLoss,
    MSERegressionLoss,
    SmoothCELoss,
    SmoothL1RegressionLoss,
)
from .lovasz import LovaszLoss
from .instance import InstanceSegmentationLoss
from .instance_fast import FastInstanceSegmentationLoss
from .instance_regression_fast import FastInstanceSegmentationRegressionLoss
