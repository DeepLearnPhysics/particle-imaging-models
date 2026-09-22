"""The opt-in adapter changes execution, not training/history policy."""

import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from exex import xm_cluster as xc

from pimm.launch import _exex
from pimm.launch.config import finalize_config
from pimm.launch.schema import LaunchConfig


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "configs").mkdir(parents=True)
    (root / "configs/train.py").write_text("epoch = 1\n")
    monkeypatch.setattr(_exex, "ROOT", root)
    monkeypatch.setattr(
        _exex.subprocess, "check_output", lambda *a, **k: b"configs/train.py\0"
    )
    value = LaunchConfig().to_dict()
    value["train"]["config"] = "train"
    value["train"]["python"] = sys.executable
    value["paths"]["exp_root"] = str(tmp_path / "experiments")
    value["resources"].update(
        scheduler="slurm",
        nproc_per_node=2,
        cpus_per_proc=3,
        account="science",
        qos="preemptable",
        mem="8G",
        time="00:05:00",
    )
    value["executor"] = "batch"
    value["run"]["name"] = "probe"
    return finalize_config(
        value, launch_timestamp="2026-09-22_00-00-00", require_config=True
    )


@pytest.mark.parametrize("value,result", [(None, False), ("0", False), ("1", True)])
def test_explicit_gate(monkeypatch, value, result):
    monkeypatch.delenv("PIMM_USE_EXEX", raising=False)
    if value is not None:
        monkeypatch.setenv("PIMM_USE_EXEX", value)
    assert _exex.enabled() is result


def test_invalid_gate_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("PIMM_USE_EXEX", "typo")
    with pytest.raises(SystemExit, match="0 or 1"):
        _exex.enabled()


@pytest.mark.parametrize("command", ["launch", "submit"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_dispatch_keeps_existing_path(monkeypatch, cfg, command, enabled):
    cli = importlib.import_module("pimm.cli." + command)
    parsed = SimpleNamespace(
        launch_config_dict=lambda: cfg,
        dry_run=True,
        output="intent.json",
        no_remote=True,
    )
    monkeypatch.setattr(cli, "parse_command", lambda *a: (parsed, []))
    monkeypatch.setattr(cli, "finalize_config", lambda *a, **k: cfg)
    monkeypatch.setenv("PIMM_USE_EXEX", "1" if enabled else "0")
    new, old = Mock(return_value=0), Mock(return_value=0)
    monkeypatch.setattr(_exex, "run", new)
    monkeypatch.setattr(cli, "launch" if command == "launch" else "run_submit", old)
    assert cli.main([]) == 0
    assert new.call_count == int(enabled)
    assert old.call_count == int(not enabled)
    if enabled:
        assert new.call_args.kwargs == {"dry_run": True, "output": "intent.json"}


@pytest.mark.parametrize("interactive", [False, True])
def test_slurm_and_history_are_independent(cfg, interactive):
    cfg["interactive"] = interactive
    cfg["chain"]["jobs"] = 3
    cfg["env"]["PIMM_WANDB_HISTORY"] = "append"
    _, _, source, executor, args, env, continuation = _exex._plan(cfg)
    assert executor.mode == ("salloc" if interactive else "sbatch")
    assert executor.resources == {
        "account": "science",
        "qos": "preemptable",
        "mem": "8G",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 6,
        "gres": "gpu:2",
    }
    assert executor.walltime.total_seconds() == 300
    assert continuation == xc.Continuation(
        checkpoint="checkpoint", max_attempts=3, pause_before=120
    )
    assert "pimm/launch/torchrun.sh" in source.entrypoint.commands[-1]
    assert "pimm.train" in source.entrypoint.commands[-1]
    assert not any("wandb_fork" in repr(arg) for arg in args)
    assert env["PIMM_WANDB_HISTORY"] == "append"
    assert env["PIMM_USE_EXEX"] == "1"


def test_remote_host_and_storage_come_from_pimm(cfg):
    cfg["submit"]["host"] = "nersc"
    cfg["submit"]["folder"] = "/pscratch/me/exex"
    _, config, _, executor, *_ = _exex._plan(cfg)
    assert config.cluster_settings(executor.cluster).hostname == "nersc"
    assert config.cluster_settings(executor.cluster).storage_root == "/pscratch/me/exex"
    assert executor.workdir_root == "/pscratch/me/exex/work"


def test_shifter_preserves_runtime_gpu_library_paths(cfg):
    cfg["container"].update(
        runtime="shifter",
        image="docker:example/train:latest",
        module="gpu,nccl-plugin",
        unset_env=["LD_LIBRARY_PATH", "LD_PRELOAD"],
    )
    _, _, package, executor, *_ = _exex._plan(cfg)
    assert isinstance(package, xc.ShifterContainer)
    assert executor.container_options.modules == ["gpu", "nccl-plugin"]
    # Clear host inheritance at entry, not paths installed by Shifter's GPU module.
    assert executor.container_options.extra_options == ["--clearenv"]
    assert not any(
        command.startswith("unset ")
        for command in package.entrypoint.entrypoint.commands
    )


def test_runtime_bootstrap_expands_attempt_path_after_arguments(cfg, tmp_path):
    _, _, source, _, args, _, _ = _exex._plan(cfg)
    commands = source.entrypoint.commands
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join([*commands[:-1], 'printf "%s\\n" "$@"']),
            "exex",
            *args,
        ],
        env={**os.environ, "EXEX_OUTPUT_DIR": str(tmp_path / "attempt with spaces")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        *args,
        f"save_path={tmp_path / 'attempt with spaces'}/checkpoint",
    ]


def test_gate_does_not_require_an_optional_checkpoint_saver(monkeypatch, cfg):
    monkeypatch.setenv("PIMM_USE_EXEX", "1")
    cfg["train"]["options"].clear()
    result = finalize_config(
        cfg, launch_timestamp=cfg["timestamp"], require_config=True
    )
    assert "hooks.CheckpointSaverIteration.backend" not in result["train"]["options"]


def test_dry_run_does_not_stage_or_submit(cfg, monkeypatch, capsys):
    submit = Mock(side_effect=AssertionError("unexpected staging/submission"))
    monkeypatch.setattr(xc, "create_experiment", submit)
    cfg["env"]["WANDB_API_KEY"] = "do-not-print"
    assert _exex.run(cfg, dry_run=True) == 0
    captured = capsys.readouterr().out
    assert "do-not-print" not in captured
    assert json.loads(captured)["env"]["WANDB_API_KEY"] == "<redacted>"
    assert not (_exex.ROOT / ".exex").exists()


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("chain", "resume_first", True, "--train.weight"),
        ("watchdog", "interval_min", 1, "no watchdog"),
        ("submit", "secret_env", {"TOKEN": "secret"}, "secret_env"),
    ],
)
def test_unsupported_settings_fail_before_submission(cfg, section, key, value, message):
    cfg[section][key] = value
    with pytest.raises(SystemExit, match=message):
        _exex.run(cfg, dry_run=True)


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("backend", ["static", "c10d"])
@pytest.mark.parametrize("explicit_endpoint", [False, True])
def test_multinode_reuses_bootstrap_and_single_continuation(
    cfg, tmp_path, interactive, backend, explicit_endpoint
):
    cfg["resources"]["nnodes"] = 2
    cfg["interactive"] = interactive
    cfg["chain"]["jobs"] = 2
    cfg["rdzv"]["backend"] = backend
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    cfg["train"]["python"] = str(fake_python)
    _, _, source, executor, args, _, continuation = _exex._plan(cfg)
    assert executor.resources["nodes"] == executor.resources["ntasks"] == 2
    assert executor.srun_options[:3] == [
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
    ]
    assert "--overlap" in executor.srun_options
    assert continuation.max_attempts == 2
    result = subprocess.run(
        ["bash", "-e", "-c", "\n".join(source.entrypoint.commands), "worker", *args],
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in {"MASTER_ADDR", "MASTER_PORT", "PIMM_NODE_RANK"}
            },
            "EXEX_OUTPUT_DIR": str(tmp_path / "output"),
            **(
                {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29999"}
                if explicit_endpoint
                else {}
            ),
            "SLURM_JOB_ID": "42",
            "SLURM_PROCID": "1",
            "SLURM_LAUNCH_NODE_IPADDR": "127.0.0.2",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    argv = result.stdout.splitlines()
    assert "--nnodes=2" in argv and "--nproc-per-node=2" in argv
    assert ("--node-rank=1" in argv) == (backend == "static")
    endpoint = "127.0.0.1:29999" if explicit_endpoint else "127.0.0.2:20042"
    assert f"--rdzv-endpoint={endpoint}" in argv
    assert argv[-len(args) - 1 :] == [
        *args,
        f"save_path={tmp_path / 'output'}/checkpoint",
    ]


def test_local_cannot_allocate_multiple_nodes(cfg):
    cfg["resources"].update(scheduler="local", nnodes=2)
    with pytest.raises(SystemExit, match="single-node Local"):
        _exex._plan(cfg)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    from exex import execution

    from pimm.engines._execution import configure_training

    class HookBase:
        def after_train(self):
            pass

    output = tmp_path / "output"
    monkeypatch.setenv("PIMM_USE_EXEX", "1")
    monkeypatch.setenv("EXEX_OUTPUT_DIR", str(output))
    monkeypatch.setenv("EXEX_PAUSE_REQUEST", str(tmp_path / "pause"))
    events = []
    registry = SimpleNamespace(register_module=Mock())
    comm = SimpleNamespace(
        is_main_process=lambda: True,
        all_gather=lambda value: [value],
        synchronize=lambda: events.append("barrier"),
    )
    manager = SimpleNamespace(save_epoch_checkpoint=lambda **k: events.append("save"))
    retain = Mock()
    modules = {
        "pimm.engines.hooks.builder": SimpleNamespace(HOOKS=registry),
        "pimm.engines.hooks.default": SimpleNamespace(HookBase=HookBase),
        "pimm.utils": SimpleNamespace(comm=comm),
        "pimm.utils.checkpoints": SimpleNamespace(
            CheckpointManager=lambda owner: manager, publish_full_state_snapshot=retain
        ),
        "pimm.utils.path": SimpleNamespace(
            latest_complete_checkpoint=lambda path: path / "last"
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(execution, "pause_requested", lambda: True)
    monkeypatch.setattr(execution, "mark_paused", lambda: events.append("marked"))
    cfg = SimpleNamespace(save_path="unchanged", resume=False, hooks=[])
    return SimpleNamespace(
        configure=configure_training,
        cfg=cfg,
        output=output,
        registry=registry,
        comm=comm,
        events=events,
        retain=retain,
        manager=manager,
    )


def test_unmanaged_worker_is_untouched(worker, monkeypatch):
    monkeypatch.delenv("EXEX_OUTPUT_DIR")
    assert not worker.configure(worker.cfg)
    assert worker.cfg.save_path == "unchanged" and worker.cfg.hooks == []


def test_worker_restores_named_artifact_and_installs_hook_last(
    worker, tmp_path, monkeypatch
):
    input_dir = tmp_path / "inputs"
    (input_dir / "checkpoint/model/last").mkdir(parents=True)
    (input_dir / "checkpoint/resolved_config.json").write_text('{"seed": 73}')
    monkeypatch.setenv("EXEX_INPUT_DIR", str(input_dir))
    worker.cfg.hooks = [{"type": "ModelHook"}]
    worker.cfg.seed = 999
    assert worker.configure(worker.cfg)
    assert worker.cfg.seed == 73
    assert worker.cfg.resume
    assert worker.cfg.weight == str(input_dir / "checkpoint/model/last")
    assert worker.cfg.save_path == str(worker.output / "checkpoint")
    assert [hook["type"] for hook in worker.cfg.hooks] == [
        "CheckpointLoader",
        "ModelHook",
        "_ExecutionCheckpoint",
    ]


@pytest.mark.parametrize(
    "epoch,iteration,expected",
    [(0, 0, ["save", "barrier", "marked", "pause"]), (0, 3, []), (1, 0, [])],
)
def test_safe_point_orders_checkpoint_before_marker(worker, epoch, iteration, expected):
    worker.configure(worker.cfg)
    hook = worker.registry.register_module.call_args.kwargs["module"]()
    hook.trainer = SimpleNamespace(
        train_state=SimpleNamespace(epoch=epoch),
        max_epoch=1,
        global_step=2,
        comm_info={"iter": iteration, "iter_per_epoch": 4},
        request_pause=lambda: worker.events.append("pause"),
    )
    hook.after_step()
    assert worker.events == expected
    worker.events.clear()
    hook.after_epoch()
    assert worker.events == (
        ["save"] if epoch == 1 else ["save", "barrier", "marked", "pause"]
    )


def test_final_checkpoint_precedes_final_evaluation_and_does_not_mark_pause(worker):
    worker.configure(worker.cfg)
    hook = worker.registry.register_module.call_args.kwargs["module"]()
    hook.trainer = SimpleNamespace(
        global_step=4,
        train_state=SimpleNamespace(epoch=1),
        max_epoch=1,
        model=SimpleNamespace(weight="final"),
    )
    saved = []
    worker.manager.save_epoch_checkpoint = lambda **kwargs: saved.append(
        (hook.trainer.model.weight, kwargs["step_count"])
    )

    hook.after_epoch()
    # FinalEvaluator(test_last=False) replaces this shared model's weights.
    hook.trainer.model.weight = "best"
    hook.after_train()

    assert saved == [("final", 4)]
    assert worker.events == []


def test_already_completed_resume_retains_input_without_resaving(worker, tmp_path):
    source = tmp_path / "last"
    source.mkdir()
    worker.cfg.resume, worker.cfg.weight = True, str(source)
    worker.configure(worker.cfg)
    hook = worker.registry.register_module.call_args.kwargs["module"]()
    hook.trainer = SimpleNamespace(
        cfg=worker.cfg, _training_already_complete=lambda: True
    )
    hook.before_train()
    worker.retain.assert_called_once_with(
        source, worker.output / "checkpoint/model/last"
    )
    assert worker.events == []


@pytest.mark.parametrize("main_process", [False, True])
def test_resume_copies_best_weights_without_aliasing_the_input(
    worker, tmp_path, monkeypatch, main_process
):
    source = tmp_path / "input" / "model" / "last"
    source.mkdir(parents=True)
    best = source.parent / "model_best.pth"
    best.write_bytes(b"previous best")
    worker.cfg.resume, worker.cfg.weight = True, str(source)
    worker.configure(worker.cfg)
    monkeypatch.setattr(worker.comm, "is_main_process", lambda: main_process)
    hook = worker.registry.register_module.call_args.kwargs["module"]()
    hook.trainer = SimpleNamespace(
        cfg=worker.cfg, _training_already_complete=lambda: False
    )

    hook.before_train()

    retained = worker.output / "checkpoint/model/model_best.pth"
    assert retained.exists() == main_process
    if main_process:
        assert retained.read_bytes() == b"previous best"
        assert not os.path.samefile(best, retained)
        retained.write_bytes(b"new best")
        assert best.read_bytes() == b"previous best"
    worker.retain.assert_not_called()
