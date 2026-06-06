"""Public dataset package surface.

Importing this package registers detector datasets and transforms with the
global registries used by config-driven training jobs.
"""

from .defaults import DefaultDataset, ConcatDataset
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn, inseg_collate_fn
from .stateful import (
    StatefulRandomSampler,
    set_dataloader_epoch,
    dataloader_state_dict,
    load_dataloader_state_dict,
)

# Detector and physics datasets.
from .pilarnet import PILArNetH5Dataset
from .jaxtpc_dataset import JAXTPCDataset
from .lucid_dataset import LUCiDDataset
from . import transform  # register transform classes

# Dataloader adapters.
from .dataloader import MultiDatasetDataloader
