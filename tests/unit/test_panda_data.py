import random

import h5py
import numpy as np
import pytest

from pimm.datasets.pilarnet import PILArNetH5Dataset
from pimm.datasets.pilarnet.decode import decode_event
from pimm.datasets.transform import FourierTime, GridSample, JointLabel, NEXTTopologyCut


@pytest.fixture
def pilarnet_v2_root(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    with h5py.File(train / "events.h5", "w") as handle:
        columns = {
            key: handle.create_dataset(key, (3,), dtype=h5py.vlen_dtype(np.float32))
            for key in ("point", "cluster", "cluster_extra")
        }
        for i, count in enumerate((4, 8, 12)):
            points = np.zeros((count, 8), np.float32)
            points[:, 0] = np.arange(count) + i * 20
            points[:, 3] = 1
            columns["point"][i] = points.ravel()
            columns["cluster"][i] = np.array([count, 0, 0, 0, 1, 2], np.float32)
            columns["cluster_extra"][i] = np.array([0, 2, 3, 4, 5], np.float32)
    return tmp_path


def test_v2_uses_the_leading_cluster_momentum():
    points = np.zeros((5, 8), np.float32)
    points[:, 3] = 1
    clusters = np.array([[2, 7, 7, 0, 0, 1], [3, 8, 7, 0, 0, 1]])
    extra = np.array([[0, 9, 1, 2, 3], [0, 4, 1, 2, 3]])
    result = decode_event(points, clusters, extra, revision="v2")
    np.testing.assert_array_equal(result["momentum"], np.full((5, 1), 9))
    np.testing.assert_array_equal(result["instance_particle"], np.zeros((5, 1)))
    with pytest.raises(ValueError, match="width 5"):
        decode_event(points, clusters, np.zeros((2, 6)), revision="v2")


def test_point_limits_and_seeded_subset(pilarnet_v2_root):
    options = dict(data_root=str(pilarnet_v2_root), revision="v2", min_points=0, transform=[])
    limited = PILArNetH5Dataset(**dict(options, min_points=4), max_points=8)
    assert len(limited) == 2
    assert [len(limited[i]["coord"]) for i in range(2)] == [4, 8]
    first = PILArNetH5Dataset(**options, max_len=2, subset_seed=3)
    second = PILArNetH5Dataset(**options, max_len=2, subset_seed=3)
    assert [first.get_data_name(i) for i in range(2)] == [second.get_data_name(i) for i in range(2)]
    assert first._num_source_events() == 2


def test_overlay_donor_subset_is_opt_in(pilarnet_v2_root):
    options = dict(
        data_root=str(pilarnet_v2_root), revision="v2", min_points=0, transform=[],
        max_len=1, overlay_n_events=2, overlay_rotate_90=False, overlay_dedup=False,
        overlay_max_instances=dict(instance_particle=8),
    )
    legacy = PILArNetH5Dataset(**options)
    closed = PILArNetH5Dataset(**options, overlay_within_subset=True)
    random.seed(0)
    assert len(legacy[0]["coord"]) > 4
    random.seed(0)
    assert len(closed[0]["coord"]) == 4


def test_overlay_hosts_can_omit_optional_controls(pilarnet_v2_root):
    dataset = PILArNetH5Dataset(
        data_root=str(pilarnet_v2_root), revision="v2", min_points=0,
        transform=[], overlay_n_events=2,
    )
    for name in ("overlay_rotate_90", "overlay_dedup", "overlay_max_instances"):
        delattr(dataset, name)
    random.seed(0)
    sample = dataset[0]
    assert len(sample["coord"]) > 0
    assert sample["name"].endswith("_overlay2")


def test_deterministic_grid_preserves_sums_and_rng():
    sample = dict(
        coord=np.array([[0.1, 0, 0], [0.2, 0, 0], [1.1, 0, 0], [1.2, 0, 0]]),
        energy=np.arange(1, 5, dtype=np.float32)[:, None],
    )
    transform = GridSample(grid_size=1, mode="deterministic", return_inverse=True, sum_keys=("energy",))
    outputs = []
    for seed in (1, 42):
        np.random.seed(seed)
        state = np.random.get_state()
        outputs.append(transform({key: value.copy() for key, value in sample.items()}))
        current = np.random.get_state()
        assert state[0] == current[0] and state[2:] == current[2:]
        np.testing.assert_array_equal(state[1], current[1])
    np.testing.assert_array_equal(outputs[0]["coord"], outputs[1]["coord"])
    np.testing.assert_array_equal(outputs[0]["energy"][outputs[0]["inverse"]].ravel(), [3, 3, 7, 7])


def test_deterministic_grid_keeps_one_labeled_point_per_voxel():
    sample = dict(
        coord=np.array([[0.1, 0, 0], [0.2, 0, 0], [1.1, 0, 0]]),
        segment=np.array([2, 3, 4]), sampled_index=np.array([1, 0]),
    )
    result = GridSample(grid_size=1, mode="deterministic")(sample)
    assert len(result["coord"]) == 2
    np.testing.assert_array_equal(result["coord"][result["sampled_index"]], [[0.1, 0, 0]])


def test_fourier_time_clock_phases():
    result = FourierTime(periods_ns=[8], subtract_min=False, include_coarse=False)(
        dict(time=np.array([0, 2, 4], np.float64))
    )
    np.testing.assert_allclose(result["time"], [[0, 1], [1, 0], [0, -1]], atol=1e-7)


def test_joint_labels_ignore_unlisted_combinations():
    result = JointLabel(keys=("motif", "pid"), combinations=((1, 2), (0, 1)))(
        dict(motif=np.array([[1], [0], [4]]), pid=np.array([[2], [1], [2]]))
    )
    np.testing.assert_array_equal(result["segment"], [[0], [1], [-1]])


def test_next_topology_rejects_multiple_tracks_and_wall_hits():
    transform = NEXTTopologyCut()
    connected = dict(coord=np.array([[0, 0, 100], [14, 0, 100], [16, 0, 100]], np.float32))
    assert transform(connected) is connected
    assert transform(dict(coord=np.array([[0, 0, 100], [100, 0, 100]], np.float32))) is None
    assert transform(dict(coord=np.array([[1801, 0, 100]], np.float32))) is None
