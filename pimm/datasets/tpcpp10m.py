"""TPCpp-10M sPHENIX TPC events read from the original release (Li et al., Zenodo 10.5281/zenodo.16970029)."""

import glob
import os

import numpy as np

from .builder import DATASETS
from .catalog import resolve_data_root
from .pilarnet.h5 import PILArNetH5Dataset

FOLDERS = {
    "train": "labeled/train",
    "validation": "labeled/validation",
    "test": "labeled/test",
    "unlabeled": "unlabeled",
}
LABELS = ("track_ids", "pid_labels", "noise_tags")


@DATASETS.register_module()
class TPCpp10MDataset(PILArNetH5Dataset):
    """TPCpp-10M events from the unzipped release under ``data_root``, in the PILArNet item format.

    The release's ``labeled/{train,validation,test}/{spacepoints,track_ids,pid_labels,noise_tags}[_NNN].npz``
    and ``unlabeled/spacepoints_NNN.npz`` hold flat per-point ``data`` arrays and, in ``spacepoints``, the
    points per event in ``size``. Labeled splits are inflated into memory when the dataset is built. The
    unlabeled shards are memory-mapped from uncompressed copies that you write once with
    ``python -m pimm.datasets.preprocessing.tpcpp10m <data_root>``.

    Each item holds the keys of :class:`~pimm.datasets.PILArNetH5Dataset` with ``revision="v2"``:

    - ``coord`` (N, 3) float32: x, y, z (``spacepoints[:, 1:4]``)
    - ``energy`` (N, 1) float32: deposited energy (``spacepoints[:, 0]``)
    - ``momentum`` (N, 1) and ``vertex`` (N, 3) float32: zeros, not in the release
    - ``segment_motif`` (N, 1) int32: 1 (track), 4 for noise points
    - ``segment_pid`` (N, 1) int32: 0 other, 1 pion, 2 kaon, 3 proton, 4 electron; 5 for unlabeled events
    - ``instance_particle`` (N, 1) int32: track ids renumbered 0..K-1 in id order; 0 for unlabeled events
    - ``instance_interaction`` (N, 1) int32: 0, -1 for noise points
    - ``segment_interaction`` (N, 1) int32: 1, 0 for noise points
    - ``name`` (``"<shard file>_<event in shard>"``), ``split`` and ``revision`` (``"v2"``)

    A labeled event's points are ordered by a stable sort on track id, so each track's points are contiguous.

    Args:
        data_root (str): The release's root folder; when it does not exist, ``$PIMM_DATA_ROOT/tpcpp-10m``, then
            ``~/.cache/pimm/datasets/tpcpp-10m``. Defaults to ``"data/tpcpp-10m"``.
        split ({"train", "validation", "test", "unlabeled"}): Release split. Defaults to ``"train"``.
        transform (list[dict] | None): Transform configs; ``None`` uses
            :func:`~pimm.datasets.pilarnet.default_transform`. Defaults to ``None``.
        test_mode (bool): Unsupported; must be ``False``. Defaults to ``False``.
        loop (int): Epoch multiplier. Defaults to ``1``.
        min_points (int): Events with fewer points are skipped. Defaults to ``1024``.
        max_points (int | None): Events with more points are skipped. Defaults to ``None``.
        max_len (int): Cap on the number of events before ``loop`` (-1 = no cap). Defaults to ``-1``.
        subset_seed (int | None): With ``max_len``, draw the events as
            :class:`~pimm.datasets.PILArNetH5Dataset` does. Defaults to ``None``.
        overlay_n_events, overlay_prob, overlay_rotate_90, overlay_dedup, overlay_max_instances: Event overlay,
            as in :class:`~pimm.datasets.PILArNetH5Dataset`. Rotation and deduplication are meant for the LArTPC
            voxel grid and default to ``False``.
    """

    def __init__(
        self,
        data_root="data/tpcpp-10m",
        split="train",
        transform=None,
        test_mode=False,
        loop=1,
        min_points=1024,
        max_points=None,
        max_len=-1,
        subset_seed=None,
        overlay_n_events=1,
        overlay_prob=1.0,
        overlay_rotate_90=False,
        overlay_dedup=False,
        overlay_max_instances=None,
    ):
        data_root = resolve_data_root("tpcpp-10m", data_root)
        self.folder = os.path.join(data_root, FOLDERS[split])
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            test_mode=test_mode,
            loop=loop,
            energy_threshold=0.0,
            min_points=min_points,
            max_len=max_len,
            revision="v2",
            overlay_n_events=overlay_n_events,
            overlay_prob=overlay_prob,
            overlay_rotate_90=overlay_rotate_90,
            overlay_dedup=overlay_dedup,
            overlay_max_instances=overlay_max_instances,
            max_points=max_points,
            subset_seed=subset_seed,
        )
        self.offsets = [np.concatenate([[0], np.cumsum(sizes)]) for sizes in self.event_sizes]
        self.arrays = [self._load(k) for k in range(len(self.files))]

    def get_files(self):
        return sorted(glob.glob(os.path.join(self.folder, "spacepoints*.npz")))

    def _event_sizes(self, path):
        with np.load(path) as f:
            return f["size"]

    def _load(self, k):
        """Shard ``k``'s spacepoints, then its track ids, pid labels and noise tags when labeled."""
        path = self.files[k]
        if self.split == "unlabeled":
            data = path.replace(".npz", ".data.npy")
            if not os.path.exists(data):
                raise FileNotFoundError(
                    f"{data} is missing; write the unlabeled shards once with "
                    f"`python -m pimm.datasets.preprocessing.tpcpp10m {os.path.dirname(self.folder)}`"
                )
            arrays = [np.load(data, mmap_mode="r")]
        else:
            arrays = []
            for key in ("spacepoints",) + LABELS:
                with np.load(path.replace("spacepoints", key)) as f:
                    arrays.append(f["data"])
        if any(len(a) != self.offsets[k][-1] for a in arrays):
            raise ValueError(f"{path}: arrays do not hold the {self.offsets[k][-1]} points of its events")
        return arrays

    def get_data(self, idx):
        k, event = self._locate(idx)
        start, end = self.offsets[k][event : event + 2]
        points, *labels = (a[start:end] for a in self.arrays[k])
        n = len(points)
        if labels:
            order = np.argsort(labels[0], kind="stable")
            points = points[order]
            track, pid, noise = (a[order] for a in labels)
            signal = ~noise
            instance = np.unique(track, return_inverse=True)[1]
        else:
            pid, signal, instance = np.full(n, 5), np.ones(n, bool), np.zeros(n)
        return dict(
            coord=points[:, 1:].astype(np.float32),
            energy=points[:, :1].astype(np.float32),
            momentum=np.zeros((n, 1), np.float32),
            vertex=np.zeros((n, 3), np.float32),
            segment_motif=np.where(signal, 1, 4).astype(np.int32)[:, None],
            segment_pid=pid.astype(np.int32)[:, None],
            instance_particle=instance.astype(np.int32)[:, None],
            instance_interaction=np.where(signal, 0, -1).astype(np.int32)[:, None],
            segment_interaction=signal.astype(np.int32)[:, None],
            name=self.get_data_name(idx),
            split=self.split,
            revision=self.revision,
        )
