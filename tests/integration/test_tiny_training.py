import os
from pathlib import Path
import sys
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_loss_goes_down,
    check_overfit,
    strip_escape_codes,
    training_losses,
)


pytestmark = [pytest.mark.gpu, pytest.mark.external_data]
ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 300


def _env(pilarnet_mini_root):
    return {
        "HF_HUB_DISABLE_XET": "1",
        "PILARNET_DATA_ROOT_V2": str(pilarnet_mini_root),
        "UV_PROJECT_ENVIRONMENT": os.environ.get(
            "UV_PROJECT_ENVIRONMENT",
            sys.prefix,
        ),
        "WANDB_MODE": "disabled",
    }


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("tiny-training")


@pytest.fixture(scope="module")
def tiny_training_process(
    run_process: Callable[..., ProcessResult],
    pilarnet_mini_root,
    output_dir,
):
    command = [
        "uv", "run", "--no-sync", "pimm", "launch",
        "--site", "local",
        "--paths.repo-root", str(ROOT),
        "--paths.exp-root", str(output_dir),
        "--resources.nproc-per-node", "1",
        "--resources.cpus-per-proc", "2",
        "--run.name", "tiny-semseg",
        "--run.no-timestamp",
        "--train.config", "tests/tiny_semseg",
        "--train.no-code-copy",
        "--",
        "data.train.data_root=" + str(pilarnet_mini_root),
        "data.val.data_root=" + str(pilarnet_mini_root),
    ]
    return run_process(
        command,
        env=_env(pilarnet_mini_root),
        timeout=TIMEOUT,
    )


@pytest.fixture(scope="module")
def overfit_process(
    run_process: Callable[..., ProcessResult],
    pilarnet_mini_root,
    output_dir,
):
    command = [
        "uv", "run", "--no-sync", "pimm", "launch",
        "--site", "local",
        "--paths.repo-root", str(ROOT),
        "--paths.exp-root", str(output_dir),
        "--resources.nproc-per-node", "1",
        "--resources.cpus-per-proc", "2",
        "--run.name", "tiny-overfit",
        "--run.no-timestamp",
        "--train.config", "tests/tiny_semseg",
        "--train.no-code-copy",
        "--",
        "epoch=100",
        "data.train.data_root=" + str(pilarnet_mini_root),
        "data.train.max_len=4",
        "data.val.data_root=" + str(pilarnet_mini_root),
        "data.val.split=train",
        "data.val.max_len=4",
        "batch_size=1",
        "batch_size_val=4",
        "optimizer.lr=0.01",
        "optimizer.weight_decay=0.0",
        "scheduler.max_lr=0.01",
    ]
    return run_process(
        command,
        env=_env(pilarnet_mini_root),
        timeout=TIMEOUT,
    )


def _log_lines(output_dir, run_name):
    path = output_dir / "tests" / run_name / "train.log"
    return strip_escape_codes(path.read_text(encoding="utf-8")).splitlines()


def test_tiny_training_no_error(tiny_training_process, output_dir):
    assert tiny_training_process.returncode == 0, tiny_training_process
    run_dir = output_dir / "tests" / "tiny-semseg"
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "model" / "last" / "weights.pth").is_file()
    assert "Val result:" in (run_dir / "train.log").read_text(encoding="utf-8")


def test_tiny_training_loss_goes_down(tiny_training_process, output_dir):
    assert tiny_training_process.returncode == 0, tiny_training_process
    check_loss_goes_down(_log_lines(output_dir, "tiny-semseg"))


@pytest.fixture(scope="module")
def resume_process(
    run_process: Callable[..., ProcessResult],
    pilarnet_mini_root,
    output_dir,
):
    # first run trains with SimulateCrash and dies mid-training after writing a
    # save_freq checkpoint; the resume run recovers from it and finishes epoch 2.
    crash_command = [
        "uv", "run", "--no-sync", "pimm", "launch",
        "--site", "local",
        "--paths.repo-root", str(ROOT),
        "--paths.exp-root", str(output_dir),
        "--resources.nproc-per-node", "1",
        "--resources.cpus-per-proc", "2",
        "--run.name", "tiny-resume",
        "--run.no-timestamp",
        "--train.config", "tests/tiny_semseg_crash",
        "--train.no-code-copy",
        "--",
        "data.train.data_root=" + str(pilarnet_mini_root),
        "data.val.data_root=" + str(pilarnet_mini_root),
    ]
    crashed = run_process(
        crash_command,
        env={**_env(pilarnet_mini_root), "PIMM_SIMULATE_CRASH_STEP": "8"},
        timeout=TIMEOUT,
    )
    losses_before = training_losses(_log_lines(output_dir, "tiny-resume"))

    resume_command = [
        "uv", "run", "--no-sync", "pimm", "launch",
        "--site", "local",
        "--paths.repo-root", str(ROOT),
        "--paths.exp-root", str(output_dir),
        "--resources.nproc-per-node", "1",
        "--resources.cpus-per-proc", "2",
        "--run.name", "tiny-resume",
        "--run.no-timestamp",
        "--train.config", "tests/tiny_semseg_crash",
        "--train.no-code-copy",
        "--train.resume",
        "--",
        "data.train.data_root=" + str(pilarnet_mini_root),
        "data.val.data_root=" + str(pilarnet_mini_root),
    ]
    resumed = run_process(
        resume_command,
        env=_env(pilarnet_mini_root),
        timeout=TIMEOUT,
    )
    return crashed, resumed, losses_before


def test_tiny_resume_crash_then_recovers(resume_process, output_dir):
    crashed, resumed, _ = resume_process
    assert crashed.returncode != 0, "first run was expected to crash"
    assert resumed.returncode == 0, resumed
    run_dir = output_dir / "tests" / "tiny-resume"
    assert (run_dir / "model" / "last" / "weights.pth").is_file()
    assert "Resuming" in (run_dir / "train.log").read_text(encoding="utf-8")


def test_tiny_resume_precrash_loss_goes_down(resume_process):
    _, _, losses_before = resume_process
    assert losses_before[-1] < losses_before[0], (
        f"pre-crash loss did not decrease: {losses_before[0]} -> {losses_before[-1]}"
    )


def test_tiny_resume_loss_not_worse(resume_process, output_dir):
    crashed, resumed, losses_before = resume_process
    assert resumed.returncode == 0, resumed
    # train.log is appended on resume, so the last loss is the resumed run's
    losses = training_losses(_log_lines(output_dir, "tiny-resume"))
    assert losses[-1] <= losses_before[-1], (
        f"resumed loss {losses[-1]} exceeded pre-crash loss {losses_before[-1]}"
    )


def test_tiny_overfit_no_error(overfit_process):
    assert overfit_process.returncode == 0, overfit_process


def test_tiny_overfit_converges(overfit_process, output_dir):
    assert overfit_process.returncode == 0, overfit_process
    check_overfit(_log_lines(output_dir, "tiny-overfit"))


def _requires_two_gpus():
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("distributed training needs at least 2 GPUs")


@pytest.fixture(scope="module")
def distributed_process(
    run_process: Callable[..., ProcessResult],
    pilarnet_mini_root,
    output_dir,
):
    _requires_two_gpus()
    command = [
        "uv", "run", "--no-sync", "pimm", "launch",
        "--site", "local",
        "--paths.repo-root", str(ROOT),
        "--paths.exp-root", str(output_dir),
        "--resources.nproc-per-node", "2",
        "--resources.cpus-per-proc", "2",
        "--run.name", "tiny-distributed",
        "--run.no-timestamp",
        "--train.config", "tests/tiny_semseg",
        "--train.no-code-copy",
        "--",
        "data.train.data_root=" + str(pilarnet_mini_root),
        "data.val.data_root=" + str(pilarnet_mini_root),
        "batch_size=2",
        "batch_size_val=2",
        "batch_size_test=2",
    ]
    return run_process(
        command,
        env=_env(pilarnet_mini_root),
        timeout=TIMEOUT,
    )


@pytest.mark.distributed
def test_tiny_distributed_no_error(distributed_process, output_dir):
    assert distributed_process.returncode == 0, distributed_process
    run_dir = output_dir / "tests" / "tiny-distributed"
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "model" / "last").is_dir()
    assert "Val result:" in (run_dir / "train.log").read_text(encoding="utf-8")


@pytest.mark.distributed
def test_tiny_distributed_loss_goes_down(distributed_process, output_dir):
    assert distributed_process.returncode == 0, distributed_process
    check_loss_goes_down(_log_lines(output_dir, "tiny-distributed"))
