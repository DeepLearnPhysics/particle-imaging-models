import pytest

from tests.conftest import PILARNET_MINI_FILES


pytestmark = pytest.mark.external_data


def test_pilarnet_mini_event_counts_and_truth(pilarnet_mini_root):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    semantic_ids = set()
    particle_ids = set()

    for filename, (_, expected_events) in PILARNET_MINI_FILES.items():
        with h5py.File(pilarnet_mini_root / filename, "r") as data:
            assert {"point", "cluster", "cluster_extra"} <= set(data)
            assert data["point"].shape[0] == expected_events
            assert data["cluster"].shape[0] == expected_events
            assert data["cluster_extra"].shape[0] == expected_events

            for raw_cluster, raw_extra in zip(
                data["cluster"],
                data["cluster_extra"],
            ):
                cluster = np.asarray(raw_cluster).reshape(-1, 6)
                cluster_extra = np.asarray(raw_extra).reshape(-1, 6)
                assert cluster.shape[0] == cluster_extra.shape[0]
                assert cluster.shape[0] > 0
                semantic_ids.update(cluster[:, -2].astype(int))
                particle_ids.update(cluster[:, -1].astype(int))

    assert semantic_ids == {0, 1, 2, 3, 4}
    assert {-1, 0, 1, 2, 3, 4} <= particle_ids


@pytest.mark.parametrize("split,expected_events", (("train", 80), ("val", 20), ("test", 20)))
def test_pilarnet_mini_dataset_splits(
    pilarnet_mini_root,
    split,
    expected_events,
):
    from pimm.datasets.pilarnet import PILArNetH5Dataset

    dataset = PILArNetH5Dataset(
        data_root=str(pilarnet_mini_root),
        revision="v3",
        split=split,
        transform=[],
        min_points=0,
    )
    assert len(dataset) == expected_events

    sample = dataset.get_data(0)
    required = {
        "coord",
        "energy",
        "segment_motif",
        "segment_pid",
        "instance_particle",
        "instance_interaction",
        "is_primary",
    }
    assert required <= set(sample)
    assert sample["coord"].shape[0] == sample["segment_motif"].shape[0]
    assert sample["coord"].shape[0] == sample["segment_pid"].shape[0]
