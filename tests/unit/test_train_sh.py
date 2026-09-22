import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("resume", [False, True], ids=["source", "old-snapshot"])
def test_train_sh_preserves_list_valued_option_as_one_argument(tmp_path, nodes, resume):
    repo = tmp_path / "repo"
    train_sh = repo / "scripts" / "train.sh"
    train_sh.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "train.sh", train_sh)
    bootstrap = repo / "pimm" / "launch" / "torchrun.sh"
    bootstrap.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "pimm" / "launch" / "torchrun.sh", bootstrap)
    config = repo / "configs" / "tests" / "launcher.py"
    config.parent.mkdir(parents=True)
    config.write_text("seed = 0\n", encoding="utf-8")

    captured_argv = tmp_path / "argv.bin"
    fake_python = tmp_path / "capture-python"
    fake_python.write_text(
        '#!/bin/sh\n'
        'if [ "$2" = "pimm.utils.path" ]; then\n'
        '  printf \'%s\\n\' "$FAKE_CHECKPOINT"\n'
        'else\n'
        '  printf \'%s\\0\' "$@" > "$CAPTURE_ARGV"\n'
        'fi\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    experiment = tmp_path / "exp" / "tests" / "argv-regression"
    if resume:
        # Older snapshots have a training payload, but no shared bootstrap.
        snapshot = experiment / "code" / "pimm"
        snapshot.mkdir(parents=True)
        (snapshot / "train.py").touch()
        (experiment / "config.py").write_text("seed = 1\n", encoding="utf-8")

    wandb_tags = "wandb_tags=['first label', 'second label']"
    env = os.environ.copy()
    env.update(
        CAPTURE_ARGV=str(captured_argv),
        EXP_ROOT=str(tmp_path / "exp"),
        MODEL_DIR="",
        FAKE_CHECKPOINT=str(experiment / "model" / "last"),
    )
    if nodes == 2:
        env.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29999",
            SLURM_PROCID="1",
            PIMM_RDZV_BACKEND="static",
        )
    subprocess.run(
        [
            "sh",
            str(train_sh),
            *(["-r", "true"] if resume else ["-C"]),
            "-p",
            str(fake_python),
            "-g",
            "1",
            "-m",
            str(nodes),
            "-c",
            "tests/launcher",
            "-n",
            "argv-regression",
            "--",
            "--options",
            wandb_tags,
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    argv = captured_argv.read_bytes().rstrip(b"\0").split(b"\0")
    assert [arg.decode() for arg in argv[-2:]] == ["--options", wandb_tags]
    if nodes == 2:
        assert b"--nnodes=2" in argv and b"--node-rank=1" in argv
    if resume:
        assert str(experiment / "code" / "pimm" / "train.py").encode() in argv
        config_arg = argv.index(b"--config-file") + 1
        assert argv[config_arg] == str(experiment / "config.py").encode()
        assert b"resume=true" in argv
