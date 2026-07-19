import warnings

import pytest

from pimm.cli.submit import parse_command
from pimm.launch.compat import normalize_legacy_cli, normalize_legacy_config
from pimm.launch.config import load_yaml
from pimm.launch.schema import LaunchConfig
from pimm.launch.utils import scheduler
from pimm.utils import PimmDeprecationWarning


def test_legacy_yaml_is_normalized_once_per_source(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
resources:
  nnodes: 2
slurm:
  time: "01:00:00"
  gpu_directive: gpus-per-node
  image: example/pimm:old
  module: gpu
  additional_parameters:
    exclusive: true
container:
  runtime: shifter
  image: example/pimm:old
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_yaml(path)
        load_yaml(path)

    assert len(caught) == 1
    assert caught[0].category is PimmDeprecationWarning
    assert "pimm 0.6.0" in str(caught[0].message)
    assert "slurm" not in config
    assert config["resources"] == {
        "scheduler": "slurm",
        "nnodes": 2,
        "time": "01:00:00",
        "gpu_directive": "gpus-per-node",
        "scheduler_options": {"exclusive": True},
    }
    assert config["container"] == {
        "runtime": "shifter",
        "image": "example/pimm:old",
        "module": "gpu",
    }


def test_legacy_base_is_normalized_before_child_merge(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text('slurm:\n  time: "01:00:00"\n', encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        '_base_: base.yaml\nresources:\n  time: "02:00:00"\n',
        encoding="utf-8",
    )

    with pytest.warns(PimmDeprecationWarning):
        config = load_yaml(child)

    assert config["resources"]["scheduler"] == "slurm"
    assert config["resources"]["time"] == "02:00:00"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"resources": {"time": "01:00:00"}, "slurm": {"time": "02:00:00"}},
            "both slurm.time and resources.time",
        ),
        ({"slurm": {"made_up": True}}, "Unknown legacy slurm setting"),
        (
            {"resources": {"scheduler": "local"}, "slurm": {}},
            "legacy slurm requires resources.scheduler='slurm'",
        ),
    ],
)
def test_legacy_yaml_contract_errors(config, message):
    with pytest.raises(SystemExit, match=message):
        normalize_legacy_config(config, source="test config")


def test_legacy_cli_options_are_rewritten_with_one_warning():
    with pytest.warns(PimmDeprecationWarning, match="pimm 0.6.0") as caught:
        argv = normalize_legacy_cli(
            [
                "--slurm.time",
                "01:00:00",
                "--slurm.gpu-directive=gres",
                "--slurm.image=example/pimm:old",
            ]
        )

    assert len(caught) == 1
    assert argv == [
        "--resources.time",
        "01:00:00",
        "--resources.gpu-directive=gres",
        "--container.image=example/pimm:old",
    ]


def test_typed_cli_accepts_legacy_alias_during_transition():
    with pytest.warns(PimmDeprecationWarning, match="pimm 0.6.0"):
        command, overrides = parse_command(
            ["--site", "s3df", "--slurm.time", "01:00:00"],
            "2026-01-02_03-04-05",
        )

    assert command.resources.time == "01:00:00"
    assert overrides == []


def test_legacy_cli_conflicts_and_unknown_options_error():
    with pytest.raises(SystemExit, match="Conflicting launcher options"):
        normalize_legacy_cli(
            ["--slurm.time", "01:00:00", "--resources.time", "02:00:00"]
        )
    with pytest.raises(SystemExit, match="Unknown legacy launcher option"):
        normalize_legacy_cli(["--slurm.made-up", "value"])

    assert normalize_legacy_cli(["--", "--slurm.made-up=value"]) == [
        "--",
        "--slurm.made-up=value",
    ]


def test_canonical_schema_rejects_unknown_settings_and_schedulers():
    with pytest.raises(SystemExit, match="Unknown launch setting"):
        LaunchConfig.from_dict({"made_up": True})
    with pytest.raises(SystemExit, match="resources.made_up"):
        LaunchConfig.from_dict({"resources": {"made_up": True}})
    with pytest.raises(SystemExit, match="resources must be a mapping"):
        LaunchConfig.from_dict({"resources": []})
    with pytest.raises(SystemExit, match="resources.scheduler must be"):
        scheduler({"resources": {"scheduler": "condor"}})
