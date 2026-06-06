"""Core transform registry helpers and composition primitives."""

from .common import *


@TRANSFORMS.register_module()
class Collect(object):
    """Select model inputs and build per-sample offset tensors."""

    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """Configure output keys and feature concatenation rules.

        For example, ``Collect(keys=["coord"], feat_keys=["coord", "color"])``
        returns ``coord``, ``offset``, and ``feat``.
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        """Collect configured keys from one transformed sample."""
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

@TRANSFORMS.register_module()
class Copy(object):
    """Copy selected keys, usually to preserve originals for evaluation."""

    def __init__(self, keys_dict=None):
        """Configure source-to-destination key copies."""
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        """Copy arrays, tensors, or metadata values into destination keys."""
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict

@TRANSFORMS.register_module()
class Update(object):
    """Assign constant values into ``data_dict`` from config."""

    def __init__(self, keys_dict=None):
        """Configure key-value updates."""
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        """Apply configured key-value updates in place."""
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict

@TRANSFORMS.register_module()
class ToTensor(object):
    """Convert numpy arrays and numeric leaves to torch tensors."""

    def __call__(self, data):
        """Recursively convert supported values to tensors."""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class Compose(object):
    """Compose config-built transforms into one callable pipeline."""

    def __init__(self, cfg=None):
        """Build transforms from registry configs in order."""
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for idx, t_cfg in enumerate(self.cfg):
            try:
                self.transforms.append(TRANSFORMS.build(t_cfg))
            except Exception as exc:
                transform_type = t_cfg.get("type") if isinstance(t_cfg, Mapping) else type(t_cfg)
                raise RuntimeError(
                    f"Failed to build transform #{idx} ({transform_type}): {exc}"
                ) from exc

    def __call__(self, data_dict):
        """Run every transform sequentially on ``data_dict``."""
        for idx, t in enumerate(self.transforms):
            try:
                data_dict = t(data_dict)
            except Exception as exc:
                keys = sorted(data_dict.keys()) if isinstance(data_dict, Mapping) else []
                raise RuntimeError(
                    f"Transform #{idx} ({t.__class__.__name__}) failed. "
                    f"Available keys: {keys}. Original error: {exc}"
                ) from exc
        return data_dict
