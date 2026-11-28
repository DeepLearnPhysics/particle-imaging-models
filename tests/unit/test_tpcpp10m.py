import numpy as np
import pytest

from pimm.datasets import TPCpp10MDataset
from pimm.datasets.preprocessing.tpcpp10m import unpack


@pytest.fixture
def release(tmp_path):
    sizes = np.array([3, 2])
    points = np.arange(20, dtype=np.float32).reshape(5, 4)
    labeled = tmp_path / "labeled" / "train"
    unlabeled = tmp_path / "unlabeled"
    labeled.mkdir(parents=True)
    unlabeled.mkdir()
    columns = dict(
        spacepoints=points,
        track_ids=np.array([7, 2, 7, 4, 4], np.uint16),
        pid_labels=np.array([1, 3, 1, 4, 4], np.uint8),
        noise_tags=np.array([False, True, False, False, False]),
    )
    for key, data in columns.items():
        np.savez_compressed(labeled / f"{key}_000.npz", data=data, size=sizes)
    np.savez_compressed(unlabeled / "spacepoints_000.npz", data=points, size=sizes)
    return tmp_path


def test_labeled_events_sort_points_by_track(release):
    item = TPCpp10MDataset(data_root=str(release), min_points=0, transform=[])[0]
    np.testing.assert_array_equal(item["energy"][:, 0], [4, 0, 8])
    np.testing.assert_array_equal(item["coord"][0], [5, 6, 7])
    np.testing.assert_array_equal(item["instance_particle"][:, 0], [0, 1, 1])
    np.testing.assert_array_equal(item["segment_pid"][:, 0], [3, 1, 1])
    np.testing.assert_array_equal(item["segment_motif"][:, 0], [4, 1, 1])
    np.testing.assert_array_equal(item["instance_interaction"][:, 0], [-1, 0, 0])
    assert item["name"] == "spacepoints_000.npz_0"


def test_unlabeled_split_reads_the_preprocessed_shards(release):
    options = dict(data_root=str(release), split="unlabeled", min_points=0, transform=[])
    with pytest.raises(FileNotFoundError, match="pimm.datasets.preprocessing.tpcpp10m"):
        TPCpp10MDataset(**options)
    unpack(str(release / "unlabeled" / "spacepoints_000.npz"))
    dataset = TPCpp10MDataset(**dict(options, min_points=3))
    assert len(dataset) == 1
    item = dataset[0]
    np.testing.assert_array_equal(item["energy"][:, 0], [0, 4, 8])
    np.testing.assert_array_equal(item["segment_pid"][:, 0], [5, 5, 5])
    np.testing.assert_array_equal(item["instance_particle"][:, 0], [0, 0, 0])
