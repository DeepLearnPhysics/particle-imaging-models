"""Retained full-state snapshots, keep-last-N pruning, and cadence validation."""

import os
from pathlib import Path
import shutil
import types

import pytest

from pimm.engines.hooks.checkpoint import CheckpointSaver, _validate_retention
from pimm.utils import checkpoints
from pimm.utils.config import Config
from pimm.utils.checkpoints import (
    full_state_snapshot_path,
    prune_retained_checkpoints,
    publish_full_state_snapshot,
)


def _dcp_checkpoint(directory, payload=b"shard"):
    """Write a minimal published split checkpoint directory."""
    os.makedirs(directory, exist_ok=True)
    for name in ("weights.pth", "trainer.dcp", ".complete"):
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(payload)
    return directory


def _names(model_dir):
    return sorted(os.listdir(model_dir))


def test_snapshot_path_names_the_step():
    assert full_state_snapshot_path("/m", 10000).endswith("/step_10000")
    assert full_state_snapshot_path("/m", 10000, legacy=True).endswith(
        "/step_10000.pth"
    )


@pytest.mark.parametrize(
    "recipe,evaluator_cadence",
    [
        ("panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py", 1000),
        ("panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8.py", 1000),
        ("polarmae/pretrain-polarmae-pilarnet.py", None),
    ],
)
def test_pretraining_recipes_keep_rolling_checkpoints(recipe, evaluator_cadence):
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(str(root / "configs" / recipe))
    savers = [hook.to_dict() for hook in cfg.hooks if hook.type == "CheckpointSaver"]
    assert len(savers) == 1
    options = savers[0]
    options.pop("type")
    saver = CheckpointSaver(**options)
    assert saver.save_freq == 1000
    assert saver.save_iter_checkpoints is False
    assert saver.save_full_state_snapshots is False
    assert saver.evaluator_every_n_steps == evaluator_cadence
    evaluators = [hook for hook in cfg.hooks if hook.type == "PretrainEvaluator"]
    assert [hook.every_n_steps for hook in evaluators] == (
        [] if evaluator_cadence is None else [evaluator_cadence]
    )


def test_publish_hard_links_and_survives_the_next_save(tmp_path):
    """The snapshot must outlive the rolling checkpoint it was published from."""
    model_dir = tmp_path / "model"
    last = _dcp_checkpoint(str(model_dir / "last"), b"step-10000 state")
    snapshot = publish_full_state_snapshot(
        last, full_state_snapshot_path(str(model_dir), 10000)
    )
    assert os.path.isdir(snapshot)
    # Hard links, not copies: same inode, so the snapshot cost no extra bytes.
    assert (
        os.stat(os.path.join(snapshot, "trainer.dcp")).st_ino
        == os.stat(os.path.join(last, "trainer.dcp")).st_ino
    )

    replacement = _dcp_checkpoint(str(model_dir / "last.tmp"), b"step-20000 state")
    os.replace(last, str(model_dir / "last.prev"))
    os.replace(replacement, last)
    shutil.rmtree(str(model_dir / "last.prev"))
    with open(os.path.join(snapshot, "trainer.dcp"), "rb") as handle:
        assert handle.read() == b"step-10000 state"
    assert os.path.exists(os.path.join(snapshot, ".complete"))


def test_publish_replaces_a_stale_snapshot_and_its_tmp(tmp_path):
    """Re-publishing the same step (a resumed run) must not leave debris."""
    model_dir = tmp_path / "model"
    last = _dcp_checkpoint(str(model_dir / "last"), b"fresh")
    destination = full_state_snapshot_path(str(model_dir), 500)
    _dcp_checkpoint(destination, b"stale")
    _dcp_checkpoint(destination + ".tmp", b"interrupted")

    publish_full_state_snapshot(last, destination)
    with open(os.path.join(destination, "weights.pth"), "rb") as handle:
        assert handle.read() == b"fresh"
    assert not os.path.exists(destination + ".tmp")


def test_prune_keeps_the_newest_n_by_step_number(tmp_path):
    """Ordering is by encoded step, not mtime or lexicographic name."""
    model_dir = tmp_path / "model"
    # Written oldest-last on purpose: mtime order is the reverse of step order.
    for step in (100000, 90000, 20000, 10000, 9000):
        _dcp_checkpoint(str(model_dir / f"step_{step}"))
    removed = prune_retained_checkpoints(str(model_dir), 3)

    assert _names(model_dir) == sorted(["step_20000", "step_90000", "step_100000"])
    assert sorted(os.path.basename(path) for path in removed) == [
        "step_10000",
        "step_9000",
    ]


def test_prune_handles_each_family_independently(tmp_path):
    """step_* snapshots and iter_*.pth weights each get their own keep-last-N."""
    model_dir = tmp_path / "model"
    for step in (1000, 2000, 3000):
        _dcp_checkpoint(str(model_dir / f"step_{step}"))
    for step in (1000, 2000, 3000, 4000):
        (model_dir / f"iter_{step}.pth").write_bytes(b"weights only")
    (model_dir / "step_2500.pth").write_bytes(b"legacy full state")

    prune_retained_checkpoints(str(model_dir), 2)

    assert _names(model_dir) == [
        "iter_3000.pth",
        "iter_4000.pth",
        "step_2000",
        "step_2500.pth",  # only member of the legacy family, so it survives
        "step_3000",
    ]


def test_prune_never_touches_last_or_best(tmp_path):
    model_dir = tmp_path / "model"
    _dcp_checkpoint(str(model_dir / "last"))
    (model_dir / "model_best.pth").write_bytes(b"best")
    for step in (1000, 2000):
        _dcp_checkpoint(str(model_dir / f"step_{step}"))

    prune_retained_checkpoints(str(model_dir), 1)

    assert _names(model_dir) == ["last", "model_best.pth", "step_2000"]


def test_prune_removes_atomic_save_siblings(tmp_path):
    """A pruned .pth takes its .prev/.tmp siblings with it."""
    model_dir = tmp_path / "model"
    for step in (1000, 2000):
        (model_dir / f"iter_{step}.pth").parent.mkdir(parents=True, exist_ok=True)
        (model_dir / f"iter_{step}.pth").write_bytes(b"w")
    (model_dir / "iter_1000.pth.prev").write_bytes(b"w")
    (model_dir / "iter_1000.pth.tmp").write_bytes(b"w")

    prune_retained_checkpoints(str(model_dir), 1)

    assert _names(model_dir) == ["iter_2000.pth"]


def test_prune_is_a_noop_without_retention(tmp_path):
    model_dir = tmp_path / "model"
    _dcp_checkpoint(str(model_dir / "step_1000"))
    assert prune_retained_checkpoints(str(model_dir), None) == []
    assert prune_retained_checkpoints(str(model_dir), 0) == []
    assert prune_retained_checkpoints(str(tmp_path / "missing"), 5) == []
    assert _names(model_dir) == ["step_1000"]


def test_validate_retention_rejects_cadences_that_never_fire():
    with pytest.raises(ValueError, match="keep_last_n"):
        _validate_retention(1000, False, None, 0)
    with pytest.raises(ValueError, match="requires a cadence"):
        _validate_retention(None, True, None, 5)
    with pytest.raises(ValueError, match="multiple of save_freq"):
        _validate_retention(3000, True, 10000, 5)


def test_validate_retention_accepts_the_intended_configurations():
    _validate_retention(1000, True, 10000, 5)  # snapshot on every 10th save
    _validate_retention(1000, True, None, None)  # snapshot on every save
    _validate_retention(1000, False, None, 5)  # prune iter_*.pth only
    _validate_retention(None, False, None, None)  # retention off entirely


def test_validate_retention_counts_epoch_end_as_a_cadence():
    """Epoch ends bypass the step counter, so they satisfy it on their own."""
    _validate_retention(None, True, None, 5, True)  # no step cadence at all
    _validate_retention(1000, True, 0, 15, True)  # epoch ends only
    # A 0 step cadence is not a step cadence, so it cannot be the only source.
    with pytest.raises(ValueError, match="requires a cadence"):
        _validate_retention(None, True, 0, 5, False)


def _legacy_manager(tmp_path):
    """A CheckpointManager wired just enough for the legacy write path."""
    manager = checkpoints.CheckpointManager.__new__(checkpoints.CheckpointManager)
    os.makedirs(str(tmp_path / "model"), exist_ok=True)
    manager.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(save_path=str(tmp_path)),
        logger=types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        ),
    )
    return manager


def test_force_full_snapshot_publishes_off_cadence(tmp_path):
    """An epoch boundary that no step cadence can hit still gets a snapshot."""
    manager = _legacy_manager(tmp_path)
    write = dict(
        fmt="legacy",
        is_best=False,
        save_freq=1000,
        save_iter_checkpoints=False,
        save_full_state_snapshots=True,
        snapshot_freq=0,  # step cadence off: epoch ends are the only source
    )
    manager._write_checkpoint({}, step_count=41666, force_full_snapshot=True, **write)
    manager._write_checkpoint({}, step_count=42000, force_full_snapshot=False, **write)

    # The second write rotates model_last.pth.prev in; only step_* matters here.
    assert [n for n in _names(tmp_path / "model") if n.startswith("step_")] == [
        "step_41666.pth"
    ]


def test_snapshot_freq_zero_and_none_differ(tmp_path):
    """``None`` inherits save_freq (snapshot every save); ``0`` means never."""
    write = dict(
        fmt="legacy",
        is_best=False,
        step_count=1000,
        save_freq=1000,
        save_iter_checkpoints=False,
        save_full_state_snapshots=True,
    )
    inherited = _legacy_manager(tmp_path / "inherited")
    inherited._write_checkpoint({}, snapshot_freq=None, **write)
    assert "step_1000.pth" in _names(tmp_path / "inherited" / "model")

    disabled = _legacy_manager(tmp_path / "disabled")
    disabled._write_checkpoint({}, snapshot_freq=0, **write)
    assert _names(tmp_path / "disabled" / "model") == ["model_last.pth"]


def test_save_epoch_checkpoint_defaults_iter_checkpoints_to_the_cadence(monkeypatch):
    """``None`` keeps the historical ``save_freq`` coupling; ``False`` overrides it."""
    manager = checkpoints.CheckpointManager.__new__(checkpoints.CheckpointManager)
    manager.trainer = None
    seen = []
    monkeypatch.setattr(checkpoints, "build_checkpoint_payload", lambda *a, **k: {})
    monkeypatch.setattr(
        checkpoints.CheckpointManager, "_checkpoint_format", lambda self: "standard"
    )
    monkeypatch.setattr(
        checkpoints.CheckpointManager,
        "_write_checkpoint",
        lambda self, payload, **kw: seen.append(kw["save_iter_checkpoints"]),
    )

    manager.save_epoch_checkpoint(save_freq=1000)
    manager.save_epoch_checkpoint(save_freq=None)
    manager.save_epoch_checkpoint(save_freq=1000, save_iter_checkpoints=False)
    assert seen == [True, False, False]


def test_saver_forwards_the_iter_checkpoint_switch():
    """Every ``CheckpointSaver`` save path must carry the flag, not just some."""
    saver = CheckpointSaver(
        save_freq=10,
        save_iter_checkpoints=False,
        save_full_state_snapshots=True,
        snapshot_freq=20,
        keep_last_n=3,
    )
    seen = []
    saver.checkpoint_manager = types.SimpleNamespace(
        save_epoch_checkpoint=lambda **kw: seen.append(kw)
    )
    saver._pending_epoch_save = None
    saver.step_count = 9
    saver.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(evaluate=False),
        # iter_per_epoch keeps _is_last_epoch_step from deferring the save.
        comm_info={"iter": 8, "iter_per_epoch": 1000},
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
    )

    saver.after_step()  # step 10: the periodic cadence fires
    saver.after_train()  # the final save
    assert [kw["save_iter_checkpoints"] for kw in seen] == [False, False]
    assert [kw["keep_last_n"] for kw in seen] == [3, 3]

    # Deferred epoch-boundary save: same kwargs, committed from after_epoch.
    saver.trainer.comm_info = {"iter": 999, "iter_per_epoch": 1000}
    saver.step_count = 19
    saver.after_step()
    assert len(seen) == 2 and saver._pending_epoch_save is not None
    saver.after_epoch()
    assert seen[-1]["save_iter_checkpoints"] is False


def _saver_on_a_ragged_epoch(**kwargs):
    """A saver whose epoch length is not a multiple of its save cadence."""
    saver = CheckpointSaver(save_freq=1000, save_full_state_snapshots=True, **kwargs)
    saver.checkpoint_manager = types.SimpleNamespace(
        save_epoch_checkpoint=lambda **kw: None
    )
    saver._pending_epoch_save = None
    # Last step of a 41666-iter epoch: no save_freq multiple lands here, ever.
    saver.step_count = 41665
    saver.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(evaluate=False),
        comm_info={"iter": 41665, "iter_per_epoch": 41666},
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
    )
    return saver


def test_epoch_end_forces_a_save_that_no_cadence_would_have_made():
    """Without this the boundary state is never written, not merely unsnapshotted."""
    saver = _saver_on_a_ragged_epoch(snapshot_on_epoch_end=True, snapshot_freq=0)
    saver.after_step()

    pending = saver._pending_epoch_save
    assert pending is not None, "epoch boundary produced no save at all"
    assert pending["step_count"] == 41666
    assert pending["force_full_snapshot"] is True
    assert pending["snapshot_freq"] == 0


def test_epoch_end_is_opt_in():
    """Default stays as it was: a ragged boundary produces nothing."""
    saver = _saver_on_a_ragged_epoch(snapshot_freq=10000)
    saver.after_step()
    assert saver._pending_epoch_save is None
