from .builder import build_criteria

from .misc import (
    BinaryFocalLoss,
    CrossEntropyHeadLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    GeodesicDirectionLoss,
    L1RegressionLoss,
    MSERegressionLoss,
    SmoothCELoss,
    SmoothL1RegressionLoss,
)
from .lovasz import LovaszLoss

from .instance import FastUnifiedInstanceLoss, ConsistentMatchingInstanceLoss
