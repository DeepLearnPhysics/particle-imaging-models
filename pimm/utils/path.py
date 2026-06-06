# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from pathlib import Path


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = (
            suffix.lower()
            if isinstance(suffix, str)
            else tuple(item.lower() for item in suffix)
        )

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive, case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def find_vcs_root(path, markers=(".git",)):
    """Finds the root directory (including itself) of specified markers.

    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None


def checkpoint_success_file(checkpoint_dir):
    """Return the sentinel file path marking a complete DCP checkpoint."""
    return osp.join(str(checkpoint_dir), ".complete")


def is_complete_dcp_checkpoint(checkpoint_dir):
    """Return whether a DCP checkpoint directory has a success sentinel."""
    return osp.isdir(checkpoint_dir) and osp.isfile(checkpoint_success_file(checkpoint_dir))


def split_checkpoint_weight_file(checkpoint_dir):
    """Return the model-weight file path inside a split checkpoint directory."""
    return osp.join(str(checkpoint_dir), "weights.pth")


def split_checkpoint_trainer_dir(checkpoint_dir):
    """Return the trainer-state DCP path inside a split checkpoint directory."""
    return osp.join(str(checkpoint_dir), "trainer.dcp")


def is_complete_split_checkpoint(checkpoint_dir):
    """Return whether a split checkpoint has weights, trainer state, and success."""
    return (
        osp.isdir(checkpoint_dir)
        and osp.isfile(checkpoint_success_file(checkpoint_dir))
        and osp.isfile(split_checkpoint_weight_file(checkpoint_dir))
        and is_complete_dcp_checkpoint(split_checkpoint_trainer_dir(checkpoint_dir))
    )


def resolve_model_weight_file(path):
    """Resolve a direct or directory weight reference to a torch weight file."""
    if osp.isfile(path):
        return path
    candidates = [
        split_checkpoint_weight_file(path),
        osp.join(str(path), "last", "weights.pth"),
        osp.join(str(path), "model", "last", "weights.pth"),
    ]
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Weight directory must contain weights.pth, last/weights.pth, "
        f"or model/last/weights.pth: {path}"
    )


def latest_complete_checkpoint(model_save_dir):
    """Return the newest complete split, DCP, or legacy checkpoint in a model dir."""
    model_save_dir = Path(model_save_dir)
    latest: Path | None = None
    for candidate in (
        model_save_dir / "last",
        model_save_dir / "last.prev",
        model_save_dir / "model_last.pth",
    ):
        if candidate.is_dir():
            if not (
                is_complete_split_checkpoint(candidate)
                or is_complete_dcp_checkpoint(candidate)
            ):
                continue
        elif not candidate.is_file():
            continue
        if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
            latest = candidate
    return latest


def _main(argv=None):
    parser = argparse.ArgumentParser(description="pimm path helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    latest = subparsers.add_parser(
        "latest-checkpoint",
        help="Print latest complete checkpoint under a model directory",
    )
    latest.add_argument("model_save_dir")
    args = parser.parse_args(argv)

    if args.command == "latest-checkpoint":
        checkpoint = latest_complete_checkpoint(args.model_save_dir)
        if checkpoint is not None:
            print(checkpoint)
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(_main())
