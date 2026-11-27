"""Core transform registry helpers and composition primitives."""

from .common import *


@TRANSFORMS.register_module()
class Collect(object):
    """Final projection of a sample dict into the model-facing batch contract.

    Keeps only the requested keys, builds per-sample ``offset`` count tensors,
    and concatenates feature groups (e.g. ``feat_keys`` -> ``feat``) into the
    fused input tensors a backbone consumes. This is normally the last
    transform in a pipeline, run after ``ToTensor``. Reads the keys named in
    ``keys`` and in every ``*_keys`` group; the length used for ``offset`` is
    read from the source key's first dimension (``coord`` by default). Returns
    a new dict containing only the collected outputs. Registered as
    ``Collect`` -- use this string as the ``type`` in a ``transform=[...]``
    config list.

    Each ``*_keys`` keyword argument names a list of source keys; the suffix
    ``_keys`` is stripped and the named arrays are concatenated along ``dim=1``
    (cast to float) under the resulting key. For example ``feat_keys`` produces
    ``feat`` -- whose total channel width sets the backbone ``in_channels`` --
    and ``offset_keys_dict`` maps each output key to the source key whose row
    count becomes its ``[N]`` offset tensor.

    Args:
        keys (str | Sequence[str]): keys to copy through verbatim into the
            output dict. A bare string is wrapped into a single-element list.
        offset_keys_dict (dict, optional): mapping of output offset key to the
            source key whose number of rows is recorded as a length-1 tensor.
            Defaults to ``dict(offset="coord")``.
        **kwargs: feature-group specifications. Every keyword ending in
            ``_keys`` (e.g. ``feat_keys=["coord", "energy"]``) names a sequence
            of keys that are concatenated along the channel dimension into a key
            with the ``_keys`` suffix removed (e.g. ``feat``).

    Note:
        Requires that the source key behind each ``offset`` entry (``coord`` by
        default) and every key referenced in ``keys``/``*_keys`` is present.
        The concatenated feature tensor's channel width must match the model's
        ``in_channels``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import Collect, ToTensor
            >>> data = {"coord": np.array([[1., 2., 3.], [4., 5., 6.]], dtype="f4"),
            ...         "energy": np.array([[10.], [20.]], dtype="f4")}
            >>> data = ToTensor()(data)  # Collect concatenates tensors, so run after ToTensor
            >>> out = Collect(keys=("coord",), feat_keys=("coord", "energy"))(data)
            >>> sorted(out)  # only kept keys, plus the built feat / offset
            ['coord', 'feat', 'offset']
            >>> out["feat"]  # coord (3 ch) and energy (1 ch) fused -> 4 channels
            tensor([[ 1.,  2.,  3., 10.],
                    [ 4.,  5.,  6., 20.]])
            >>> out["offset"]  # number of points in this sample
            tensor([2])
    """

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
    """Duplicate keys in the sample dict under new names.

    For each ``source -> destination`` entry, deep-copies ``data_dict[source]``
    into ``data_dict[destination]`` in place. Numpy arrays are ``.copy()``-ed,
    torch tensors are ``.clone().detach()``-ed, and anything else is
    ``copy.deepcopy``-ed, so the destination is independent of the source. The
    two canonical uses are preserving a pristine copy of a key before
    augmentation mutates it (e.g. ``coord -> origin_coord`` for test-time
    re-mapping) and remapping a dataset-specific label into the conventional
    ``segment`` name (e.g. ``segment_motif -> segment``). Returns the same
    ``data_dict``. Registered as ``Copy`` -- use this string as the ``type``
    in a ``transform=[...]`` config list.

    Args:
        keys_dict (dict, optional): mapping of source key to destination key.
            Each source must already exist in the sample. Defaults to
            ``dict(coord="origin_coord", segment="origin_segment")``.

    Note:
        Requires every source key to be present in ``data_dict``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import Copy
            >>> data = {"segment_motif": np.array([2, 3, 2])}
            >>> out = Copy(keys_dict={"segment_motif": "segment"})(data)
            >>> sorted(out)  # source remains; an independent "segment" copy is added
            ['segment', 'segment_motif']
            >>> out["segment"]
            array([2, 3, 2])
    """

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
    """Inject constant key-value pairs into the sample dict.

    Writes each configured ``key: value`` straight into ``data_dict`` (in
    place, overwriting any existing entry), then returns it. Used to seed
    pipeline-control keys that downstream transforms read, such as overriding
    ``index_valid_keys`` (which keys subsampling transforms keep length-aligned)
    or declaring ``aux_position_keys`` / ``aux_direction_keys`` so geometric
    augmentations also transform auxiliary targets. Registered as ``Update`` --
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        keys_dict (dict, optional): mapping of key to constant value to assign.
            Defaults to an empty dict (no-op).

    Note:
        Values are assigned by reference; they are not copied. A common use is
        ``keys_dict={"index_valid_keys": [...]}`` to control which point-aligned
        arrays survive subsampling transforms like ``GridSample``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import Update
            >>> data = {"coord": np.zeros((2, 3), dtype="f4")}
            >>> out = Update(keys_dict={"aux_position_keys": ["primary_vertex"]})(data)
            >>> sorted(out)  # the control key is injected into the sample dict
            ['aux_position_keys', 'coord']
            >>> out["aux_position_keys"]
            ['primary_vertex']
    """

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
    """Recursively convert numpy arrays and numeric leaves to torch tensors.

    Walks the input (which may be the whole ``data_dict``, a nested mapping, or
    a sequence) and converts supported leaves to torch tensors: ``int`` becomes
    a ``LongTensor``, ``float`` becomes a ``FloatTensor``, boolean arrays are
    kept boolean, integer arrays become ``long``, and floating arrays become
    ``float``. Existing tensors and strings pass through unchanged; mappings and
    sequences are rebuilt with each element converted. Place near the end of a
    pipeline, after the numpy-based geometry and filtering steps but before
    ``Collect``. Raises ``TypeError`` on an unsupported leaf type. Registered as
    ``ToTensor`` -- use this string as the ``type`` in a ``transform=[...]``
    config list.

    Note:
        Takes no constructor arguments. Strings are detected before the generic
        sequence branch so they are left intact rather than split into
        characters.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import ToTensor
            >>> data = {"coord": np.array([[1., 2., 3.]], dtype="f4"),
            ...         "segment": np.array([2, 3]), "name": "evt0"}
            >>> out = ToTensor()(data)
            >>> type(out["coord"]).__name__, out["coord"].dtype  # float array -> float tensor
            ('Tensor', torch.float32)
            >>> type(out["segment"]).__name__, out["segment"].dtype  # int array -> long tensor
            ('Tensor', torch.int64)
            >>> out["name"]  # strings pass through unchanged
            'evt0'
    """

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
    """Build a transform pipeline from config dicts and run it in order.

    Constructed from a list of config dictionaries (the ``transform=[...]``
    list used throughout pimm). At init each entry is materialized through the
    ``TRANSFORMS`` registry via ``TRANSFORMS.build`` -- so every step's
    ``type`` must name a registered transform -- and build failures are
    re-raised with the offending step index and type. Calling the instance
    threads a single ``data_dict`` through every transform in sequence, where a
    failure is re-raised with the failing step's index, class name, and the
    keys available at that point. This is the pipeline runner itself, not a
    registered transform; do not put ``Compose`` in a config list -- pass the
    raw list of dicts and let datasets build it internally.

    Args:
        cfg (Sequence[dict], optional): ordered list of transform config dicts,
            each carrying a ``type`` plus that transform's keyword arguments.
            Defaults to an empty list (an identity pipeline).

    Note:
        Most transforms mutate and return the same ``data_dict``, but a few
        (e.g. ``GridSample`` in ``mode="test"``) return a different object such
        as a list of fragment dicts; the runner simply forwards whatever each
        step returns to the next.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform.base import Compose
            >>> pipeline = Compose([
            ...     dict(type="NormalizeCoord", center=[500., 0., 0.], scale=500.),
            ...     dict(type="ToTensor"),
            ...     dict(type="Collect", keys=("coord",), feat_keys=("coord",)),
            ... ])
            >>> data = {"coord": np.array([[0., 0., 0.], [1000., 0., 0.]], dtype="f4")}
            >>> out = pipeline(data)  # normalize -> tensor -> collect, run in order
            >>> sorted(out)
            ['coord', 'feat', 'offset']
            >>> out["coord"]  # NormalizeCoord mapped [0,1000] -> [-1,1]; ToTensor made it a tensor
            tensor([[-1.,  0.,  0.],
                    [ 1.,  0.,  0.]])
            >>> out["offset"]
            tensor([2])
    """

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


@TRANSFORMS.register_module()
class JointLabel(object):
    """Encode selected combinations of NumPy point labels as one compact target.

    ``combinations`` defines the output class order. Points whose component
    tuple is not listed are assigned ``ignore_index``. This makes the fixed
    joint target used by standalone downstream training equivalent to the
    compact joint labels produced by ``PretrainEvaluator``.

    Args:
        keys (sequence[str]): Component label keys, in tuple order.
        combinations (sequence[sequence[int]]): Allowed component tuples.
            The tuple at position ``i`` maps to output class ``i``.
        out_key (str): Destination label key. Defaults to ``"segment"``.
        ignore_index (int): Value for unlisted combinations. Defaults to ``-1``.

    Example:
        Combine a motif label and a particle-type label into one class per listed pair:

        .. code-block:: python

            dict(
                type="JointLabel",
                keys=("segment_motif", "segment_pid"),
                combinations=((0, 2), (1, 1), (2, 4)),  # (motif, pid) -> class 0, 1, 2
            )

        With ``segment_motif = [0, 1, 2, 1]`` and ``segment_pid = [2, 1, 4, 3]`` this writes
        ``segment = [0, 1, 2, -1]``: the last point's pair ``(1, 3)`` is not listed.
    """

    def __init__(
        self,
        keys,
        combinations,
        out_key="segment",
        ignore_index=-1,
    ):
        self.keys = tuple(keys)
        self.combinations = tuple(
            tuple(int(v) for v in combo) for combo in combinations
        )
        self.out_key = out_key
        self.ignore_index = int(ignore_index)
        if not self.keys:
            raise ValueError("JointLabel.keys must not be empty")
        if any(len(combo) != len(self.keys) for combo in self.combinations):
            raise ValueError("Each JointLabel combination must match len(keys)")
        if len(set(self.combinations)) != len(self.combinations):
            raise ValueError("JointLabel combinations must be unique")

    def __call__(self, data_dict):
        """Write the compact joint label while preserving component labels."""
        values = [data_dict[key] for key in self.keys]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError(
                f"JointLabel component shapes must match: "
                f"{dict(zip(self.keys, (value.shape for value in values)))}"
            )

        flat = [np.asarray(value).reshape(-1) for value in values]
        target = np.full(flat[0].shape, self.ignore_index, dtype=np.int32)
        for class_index, combo in enumerate(self.combinations):
            mask = np.ones(flat[0].shape, dtype=bool)
            for value, expected in zip(flat, combo):
                mask &= value == expected
            target[mask] = class_index

        data_dict[self.out_key] = target.reshape(reference.shape)
        return data_dict
