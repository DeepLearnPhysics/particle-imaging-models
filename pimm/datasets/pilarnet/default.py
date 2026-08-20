"""Default preprocessing preset for the PILArNet readers.

:func:`default_transform` is what a PILArNet reader uses when a config (or a
notebook) passes no ``transform`` at all. Pass ``transform=[]`` to opt out and
get raw decoded arrays instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

COORD_CENTER = (384.0, 384.0, 384.0)
COORD_SCALE = 768.0 * 3**0.5 / 2
GRID_SIZE = 1.0e-3
ENERGY_MIN = 1.0e-2 # MeV
ENERGY_MAX = 20.0   # MeV

# Truth keys collected by default. Physics targets such as ``momentum``,
# ``vertex``, and the v3_extra fields are opt-in via ``extra_keys``, since they
# widen every collated batch.
_TRUTH_KEYS = (
    "segment_motif",
    "segment_pid",
    "segment_interaction",
    "instance_particle",
    "instance_interaction",
)


def default_transform(
    copy: Mapping[str, str] | None = None,
    extra_keys: Sequence[str] = (),
) -> list[dict]:
    """Return the released-Panda preprocessing pipeline as transform configs.

    Normalizes coords about the grid center, log-scales energy, dedups voxel
    collisions with ``GridSample``, and collects ``feat = [coord, energy]``. No
    random augmentation, so it serves for inference and as a training base (add
    augmentation before ``ToTensor``).

    Args:
        copy (Mapping[str, str] | None): ``Copy`` transform ``keys_dict``, e.g.
            ``{"coord": "origin_coord"}`` to keep a pre-augmentation copy. The
            copies are collected too. Defaults to ``None``.
        extra_keys (Sequence[str]): Additional dataset keys to collect, for the
            truth this preset skips (``momentum``, ``vertex``, ``is_primary`` or
            entries in ``V3_EXTRA_KEYS``). Defaults to ``()``.

    Returns:
        list[dict]: Transform configs, ready to hand to a reader's
        ``transform=`` (a list of dicts, NOT a prebuilt ``Compose``).

    Example:
        .. code-block:: python

            >>> from pimm.datasets.pilarnet import default_transform
            >>> [t["type"] for t in default_transform()]
            ['NormalizeCoord', 'LogTransform', 'GridSample', 'ToTensor', 'Collect']
            >>> "is_primary" in default_transform(extra_keys=("is_primary",))[-1]["keys"]
            True

            Collect every v3_extra physics/lineage field with::

                from pimm.datasets.pilarnet import V3_EXTRA_KEYS
                transform = default_transform(extra_keys=V3_EXTRA_KEYS)
    """
    keys = ("coord", "grid_coord", "energy", *_TRUTH_KEYS, *extra_keys)
    if copy:
        keys = (*keys, *copy.values())
    transforms = [
        dict(type="NormalizeCoord", center=list(COORD_CENTER), scale=COORD_SCALE),
        dict(type="LogTransform", min_val=ENERGY_MIN, max_val=ENERGY_MAX),
        dict(
            type="GridSample",
            grid_size=GRID_SIZE,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
    ]
    if copy:
        transforms.append(dict(type="Copy", keys_dict=dict(copy)))
    transforms += [
        dict(type="ToTensor"),
        dict(type="Collect", keys=keys, feat_keys=("coord", "energy")),
    ]
    return transforms
