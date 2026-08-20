#!/usr/bin/env python3
"""
Download the PILArNet-M v3-family dataset from Hugging Face.

pimm reads **v3 only**: v1 has no PID/momentum/vertex columns, and v2's vertex
columns are in a different coordinate frame than ``coord``. v3 is a strict
superset of a corrected v2 -- identical points and labels, fixed vertices, plus
``is_primary`` -- so there is nothing to gain from the older branches.

The full version is ~168 GB; ``--split test`` is ~7 GB and is all the
notebooks/tutorials need. ``v3_extra`` preserves v3 and adds momentum-vector,
particle-lineage, and explicit interaction-vertex truth.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "DeepLearnPhysics/PILArNet-M"
REPO_TYPE = "dataset"
REVISIONS = ("v3", "v3_extra")


def get_shell_config_path() -> Path | None:
    """Return path to bashrc or bash_profile based on shell."""
    home = Path.home()
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        return home / ".zshrc"
    elif "bash" in shell:
        bash_profile = home / ".bash_profile"
        bashrc = home / ".bashrc"
        if bash_profile.exists():
            return bash_profile
        return bashrc
    else:
        bashrc = home / ".bashrc"
        if bashrc.exists():
            return bashrc
        return home / ".bash_profile"


def add_to_shell_config(data_root: Path, revision: str) -> None:
    """Add the revision-specific data root to the shell config."""
    env_var = f"PILARNET_DATA_ROOT_{revision.upper()}"
    config_path = get_shell_config_path()

    if config_path is None:
        print("Could not determine shell config file. Skipping environment variable setup.")
        return

    if not config_path.exists():
        print(f"Config file {config_path} does not exist. Creating it.")
        config_path.touch()

    content = config_path.read_text()
    if env_var in content:
        print(f"{env_var} already exists in {config_path}. Skipping.")
        return

    with open(config_path, "a") as f:
        f.write("\n# PILArNet dataset path\n")
        f.write(f'export {env_var}="{data_root}"\n')
    print(f"Added {env_var} to {config_path}")
    print(f"  Run 'source {config_path}' or restart your shell to use it.")


def download(
    output_dir: Path,
    revision: str = "v3",
    splits: list[str] | None = None,
) -> None:
    """Download one branch, optionally restricted to individual splits."""
    # The readers glob "*<split>/*.h5" under the data root, so keeping the
    # repo's train/val/test layout means a split-only download still resolves.
    allow_patterns = None if not splits else [f"{split}/*" for split in splits]

    print(f"\nDownloading")
    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print(f"Splits: {', '.join(splits) if splits else 'all (~168 GB)'}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download a PILArNet-M v3-family dataset from Hugging Face"
    )
    parser.add_argument(
        "--revision",
        choices=REVISIONS,
        default="v3",
        help="Dataset revision (default: v3; choose v3_extra for extended truth)",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        help="Restrict the download to one split; repeatable "
        "(default: every split, ~168 GB)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ~/.cache/pimm/pilarnet/<revision>)",
    )
    parser.add_argument(
        "--no-env-setup",
        action="store_true",
        help="Skip asking about environment variable setup",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            Path.home() / ".cache" / "pimm" / "pilarnet" / args.revision
        )

    download(args.output_dir, args.revision, args.split)

    if not args.no_env_setup:
        print("\n" + "=" * 60)
        response = (
            input(
                "Would you like to add "
                f"PILARNET_DATA_ROOT_{args.revision.upper()} to your shell config? [y/N]: "
            )
            .strip()
            .lower()
        )
        if response in ["y", "yes"]:
            add_to_shell_config(args.output_dir, args.revision)
        else:
            print("Skipping environment variable setup.")
            print("You can manually set:")
            print(
                f'  export PILARNET_DATA_ROOT_{args.revision.upper()}="{args.output_dir}"'
            )

    print("\nDownload complete")
    print(f"  {args.revision}: {args.output_dir}")


if __name__ == "__main__":
    main()
