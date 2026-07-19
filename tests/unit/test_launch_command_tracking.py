import json
from types import SimpleNamespace

from pimm.engines import train as train_engine
from pimm.engines.defaults import default_config_parser


def test_launch_command_reaches_metadata_and_wandb_notes(monkeypatch, tmp_path):
    command = "pimm submit --site nersc --train.config configs/example.py"
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.py"
    config_path.write_text(f"seed = 1\nsave_path = {str(run_dir)!r}\nresume = False\n")
    monkeypatch.setenv("PIMM_LAUNCH_COMMAND", command)

    cfg = default_config_parser(str(config_path), options=None)

    assert cfg.launch_command == command
    assert (
        json.loads((run_dir / "resolved_config.json").read_text())["launch_command"]
        == command
    )
    assert (
        json.loads((run_dir / "run_metadata.json").read_text())["launch_command"]
        == command
    )

    class FakeWriter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(train_engine, "WandbSummaryWriter", FakeWriter)
    monkeypatch.setattr(train_engine.comm, "is_main_process", lambda: True)
    cfg.use_wandb = True
    trainer = train_engine.Trainer.__new__(train_engine.Trainer)
    trainer.cfg = cfg
    trainer.logger = SimpleNamespace(info=lambda _: None)

    writer = trainer.build_writer()

    assert writer.kwargs["notes"] == command
