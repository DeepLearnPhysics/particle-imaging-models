"""The datasets that `pimm datasets download` fetches from the Hugging Face Hub, and where pimm looks for them."""

import os
from pathlib import Path

# splits: split -> Hub file patterns. revisions: pimm revision -> Hub branch, first is the default.
CATALOG = {
    "pilarnet-m": dict(
        repo="DeepLearnPhysics/PILArNet-M",
        license="Apache-2.0",
        size="168 GB",
        splits={"train": ["train/*"], "val": ["val/*"], "test": ["test/*"]},
        revisions={"v3_extra": "v3_extra", "v3": "v3", "v2": "main"},
    ),
    "tpcpp-10m": dict(
        repo="DeepLearnPhysics/tpcpp-10m",
        license="CC BY 4.0",
        size="119 GB",
        splits={
            "train": ["labeled/train/*"],
            "validation": ["labeled/validation/*"],
            "test": ["labeled/test/*"],
            "unlabeled": ["unlabeled/*"],
        },
    ),
    "wand-sk-like": dict(
        repo="DeepLearnPhysics/wand-sk-like",
        license="CC BY 4.0",
        size="328 GB",
        splits={
            "train": ["train/config_01_packed/*", "metadata/*"],
            "val": ["val/*", "metadata/*"],
            "test": ["test/*", "metadata/*"],
            "probes": ["probes/*", "metadata/*"],
        },
    ),
    # 20 tar parts; NEXTH5Dataset draws the train/holdout split
    "next-aiss": dict(repo="DeepLearnPhysics/next-aiss", license="CC BY 4.0", size="16 GB", splits={}),
}


CACHE = Path.home() / ".cache" / "pimm" / "datasets"


def data_home(name):
    """Where `pimm datasets download` puts ``name``: ``$PIMM_DATA_ROOT/<name>``, else ``~/.cache/pimm/datasets/<name>``."""
    return Path(os.environ.get("PIMM_DATA_ROOT", CACHE)) / name


def resolve_data_root(name, data_root=None):
    """The first existing folder of ``data_root``, ``$PIMM_DATA_ROOT/<name>`` and ``~/.cache/pimm/datasets/<name>``."""
    candidates = [data_root] + [Path(root) / name for root in (os.environ.get("PIMM_DATA_ROOT"), CACHE) if root]
    for path in candidates:
        if path and os.path.isdir(path):
            return str(path)
    raise FileNotFoundError(
        f"{name} is in none of {', '.join(str(p) for p in candidates if p)}; "
        f"download it with `pimm datasets download {name.replace('/', ' --revision ')}`"
    )
