#!/usr/bin/env python3
"""Export pimm checkpoints to portable pretrained model directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pimm.utils.config import Config
from pimm.utils.path import split_checkpoint_weight_file


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(run_dir: Path | None, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file() or Path(split_checkpoint_weight_file(checkpoint_path)).is_file():
        return checkpoint_path
    if run_dir is None:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    model_dir = run_dir / "model"
    candidates: list[Path] = []
    if checkpoint_path.suffix:
        candidates.append(model_dir / checkpoint)
    else:
        candidates.extend([
            model_dir / checkpoint,
            model_dir / f"{checkpoint}.pth",
            model_dir / f"{checkpoint}.safetensors",
            model_dir / f"{checkpoint}.bin",
        ])
    for candidate in candidates:
        if candidate.is_file() or Path(split_checkpoint_weight_file(candidate)).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve checkpoint {checkpoint!r} under {model_dir}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _default_config(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    config_path = run_dir / "config.py"
    return config_path if config_path.is_file() else None


def _default_training_config(run_dir: Path | None, config_path: Path | None) -> Any:
    if run_dir is not None:
        resolved = run_dir / "resolved_config.json"
        if resolved.is_file():
            return _read_json(resolved)
    if config_path is not None:
        return Config.fromfile(str(config_path))
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="last",
        help=(
            "Checkpoint path, DCP directory, or checkpoint name under <run-dir>/model. "
            "Defaults to last when --run-dir is used."
        ),
    )
    parser.add_argument("output_dir", nargs="?", help="Directory to write exported artifacts.")
    parser.add_argument("--run-dir", help="Experiment directory containing config.py and model/.")
    parser.add_argument("--config", help="Config file to use when it cannot be inferred from the run dir.")
    parser.add_argument("--model-card", help="README.md text file to include as the model card.")
    parser.add_argument("--push-to-hub", help="Optional Hugging Face Hub repo id to upload after export.")
    parser.add_argument("--public", action="store_true", help="Create a public Hub repo (default: private).")
    parser.add_argument("--token", help="Hugging Face token for upload.")
    parser.add_argument("--device", default="cpu", help="Device used while consolidating checkpoint tensors.")
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Write model.bin (torch pickle) instead of model.safetensors.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit without writing.")
    args = parser.parse_args(argv)
    if args.output_dir is None and not args.dry_run:
        parser.error("output_dir is required unless --dry-run is set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    # Bare-weights export needs no config; if one is available it's recorded as
    # training_config.json provenance, otherwise it is simply omitted.
    config_path = Path(args.config) if args.config else _default_config(run_dir)
    training_config = _default_training_config(run_dir, config_path)
    model_card = Path(args.model_card).read_text(encoding="utf-8") if args.model_card else None

    resolved = {
        "checkpoint": str(checkpoint),
        "config": str(config_path) if config_path else None,
        "output_dir": args.output_dir,
        "push_to_hub": args.push_to_hub,
        "safe_serialization": not args.no_safe_serialization,
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    # Lazy import: keeps `pimm export --help`/`--dry-run` torch-free; the ML stack
    # only loads when we actually consolidate/upload weights.
    from pimm.export import push_to_hub, save_pretrained

    output_dir = save_pretrained(
        checkpoint,
        args.output_dir,
        config_path=config_path,
        training_config=training_config,
        safe_serialization=not args.no_safe_serialization,
        model_card=model_card,
        device=args.device,
    )
    print(f"Exported pretrained model to: {output_dir}")
    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub, private=not args.public, token=args.token)
        print(f"Uploaded pretrained model to: {args.push_to_hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
