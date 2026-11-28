"""Shared input and checkpoint loading for the evaluation commands."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2) + "\n")


def load_config(path):
    from pimm.engines._train_utils import _apply_hook_overrides_from_dict
    from pimm.utils.config import Config

    cfg = Config.fromfile(str(path), import_custom_modules=False)
    if cfg.get("hooks_override"):
        _apply_hook_overrides_from_dict(cfg, cfg.hooks_override)
    return cfg


def load_model(cfg, checkpoint, device="cuda", overlays=(), disable_flash=False):
    from pimm.models import build_model
    from pimm.ops import set_attention_backend
    from pimm.utils.checkpoints import checkpoint_model_state_dict

    # The detector's reference evaluation uses PyTorch's attention.
    set_attention_backend("torch" if disable_flash else cfg.get("attention_backend", "flash"))
    model = build_model(copy.deepcopy(cfg.model)).float().to(device).eval()
    for index, path in enumerate((checkpoint, *overlays)):
        state = checkpoint_model_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing = model.load_state_dict(state, strict=index == 0)
        if missing.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint keys: {missing.unexpected_keys}")
    return model


def make_dataset(cfg, split="test", data_root=None, max_events=None):
    from pimm.datasets import build_dataset

    spec = copy.deepcopy(dict(cfg.data[split]))
    spec.update(test_mode=False, loop=1, max_len=-1 if max_events is None else max_events)
    if data_root is not None:
        spec["data_root"] = str(data_root)
    return build_dataset(spec)


def loader(dataset, indices=None, batch_size=1, workers=0, seed=0):
    from pimm.datasets import collate_fn

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                      collate_fn=collate_fn, shuffle=False, generator=generator)


def to_device(batch, device="cuda"):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch
