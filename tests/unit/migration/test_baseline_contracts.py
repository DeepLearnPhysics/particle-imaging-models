from pathlib import Path

import pytest
import torch
import yaml

from pimm.datasets.utils import collate_fn
from pimm.utils.registry import Registry


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "migration"


def _load_yaml(path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _tensor_sample(sample):
    return {
        "coord": torch.tensor(sample["coord"], dtype=torch.float32),
        "feat": torch.tensor(sample["feat"], dtype=torch.float32),
        "offset": torch.tensor(sample["offset"], dtype=torch.int64),
        "segment": torch.tensor(sample["segment"], dtype=torch.int64),
        "category": torch.tensor(sample["category"], dtype=torch.int64),
        "name": sample["name"],
        "_worker_note": sample["_worker_note"],
    }


def test_legacy_packed_batch_fixture_matches_current_collator():
    fixture = _load_yaml(FIXTURE_ROOT / "legacy_packed_batch.yaml")
    batch = collate_fn([_tensor_sample(sample) for sample in fixture["samples"]])
    expected = fixture["expected"]

    torch.testing.assert_close(
        batch["coord"], torch.tensor(expected["coord"], dtype=torch.float32)
    )
    torch.testing.assert_close(
        batch["feat"], torch.tensor(expected["feat"], dtype=torch.float32)
    )
    torch.testing.assert_close(
        batch["offset"], torch.tensor(expected["offset"], dtype=torch.int64)
    )
    torch.testing.assert_close(
        batch["segment"], torch.tensor(expected["segment"], dtype=torch.int64)
    )
    torch.testing.assert_close(
        batch["category"], torch.tensor(expected["category"], dtype=torch.int64)
    )
    assert batch["name"] == expected["name"]
    assert set(expected["omitted_keys"]).isdisjoint(batch)


def test_plain_module_fixture_has_scalar_loss_and_expected_gradient():
    fixture = _load_yaml(FIXTURE_ROOT / "plain_module_step.yaml")
    registry = Registry("synthetic_migration_models")

    @registry.register_module(fixture["registry_name"])
    class SyntheticPlainModule(torch.nn.Module):
        def __init__(self, in_features, initial_weight):
            super().__init__()
            self.projection = torch.nn.Linear(in_features, 1, bias=False)
            with torch.no_grad():
                self.projection.weight.copy_(
                    torch.tensor(initial_weight, dtype=torch.float32).unsqueeze(0)
                )

        def forward(self, batch):
            predictions = self.projection(batch["features"]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(predictions, batch["targets"])
            return {"loss": loss, "predictions": predictions}

    model = registry.build(
        {
            "type": fixture["registry_name"],
            **fixture["model"],
        }
    )
    output = model(
        {
            "features": torch.tensor(
                fixture["batch"]["features"], dtype=torch.float32
            ),
            "targets": torch.tensor(fixture["batch"]["targets"], dtype=torch.float32),
        }
    )

    assert output["loss"].ndim == 0
    assert output["loss"].item() == pytest.approx(fixture["expected"]["loss"])
    torch.testing.assert_close(
        output["predictions"],
        torch.tensor(fixture["expected"]["predictions"], dtype=torch.float32),
    )

    output["loss"].backward()
    torch.testing.assert_close(
        model.projection.weight.grad.squeeze(0),
        torch.tensor(fixture["expected"]["weight_gradient"], dtype=torch.float32),
    )


def test_migration_manifests_record_verified_baselines_and_active_checkpoints():
    status = _load_yaml(REPO_ROOT / "docs" / "migration" / "status.yaml")
    checkpoints = _load_yaml(
        REPO_ROOT / "docs" / "migration" / "public-checkpoints.yaml"
    )

    assert status["repositories"]["public"]["main_at_start"] == (
        "9491b0bf4b89bbee52a6383225a19f9c6a628a3c"
    )
    assert status["repositories"]["private"]["main_at_start"] == (
        "62239122c7a3112640743c8900cdc4336c33e59c"
    )
    assert status["repositories"]["warpconvnet"]["pinned_baseline"] == (
        "cb22e75d1b102796585bcded5f4b02a492fb7fdd"
    )
    assert status["ancestry"] == {
        "public_main_is_ancestor_of_private_main": True,
        "merge_base": "9491b0bf4b89bbee52a6383225a19f9c6a628a3c",
        "private_ahead": 110,
        "private_behind": 0,
    }

    entries = checkpoints["checkpoints"]
    assert len(entries) == 8
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert all(entry["activity"] == "ACTIVE" for entry in entries)
    assert all(entry["migration_disposition"] == "CONVERT" for entry in entries)
    assert all(entry["validation_status"] == "PENDING" for entry in entries)

    for entry in entries:
        config_paths = entry.get("configs", [entry.get("config")])
        for config_path in config_paths:
            assert config_path is not None
            assert (REPO_ROOT / config_path).is_file()


def test_milestone_zero_does_not_add_warpconvnet_to_pimm():
    project_metadata = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "warpconvnet" not in project_metadata.lower()
