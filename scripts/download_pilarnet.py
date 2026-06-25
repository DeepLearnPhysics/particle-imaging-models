#!/usr/bin/env python3
"""
Download PILArNet-M dataset from Hugging Face.

Supports downloading either v1 or v2 (or both) and optionally adds
environment variables to bashrc/bashprofile.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "DeepLearnPhysics/PILArNet-M"
REPO_TYPE = "dataset"
V1_BRANCH = "v1"
V2_TAG = "v2"


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


def add_to_shell_config(data_root_v1: Path, data_root_v2: Path) -> None:
    """Add PILARNET_DATA_ROOT_V1 and PILARNET_DATA_ROOT_V2 to shell config."""
    config_path = get_shell_config_path()
    
    if config_path is None:
        print("Could not determine shell config file. Skipping environment variable setup.")
        return
    
    if not config_path.exists():
        print(f"Config file {config_path} does not exist. Creating it.")
        config_path.touch()
    
    with open(config_path, "r") as f:
        content = f.read()
    
    v1_var = f'export PILARNET_DATA_ROOT_V1="{data_root_v1}"'
    v2_var = f'export PILARNET_DATA_ROOT_V2="{data_root_v2}"'
    
    lines_to_add = []
    if "PILARNET_DATA_ROOT_V1" not in content:
        lines_to_add.append(v1_var)
    else:
        print(f"PILARNET_DATA_ROOT_V1 already exists in {config_path}. Skipping.")
    
    if "PILARNET_DATA_ROOT_V2" not in content:
        lines_to_add.append(v2_var)
    else:
        print(f"PILARNET_DATA_ROOT_V2 already exists in {config_path}. Skipping.")
    
    if lines_to_add:
        with open(config_path, "a") as f:
            f.write("\n# PILArNet dataset paths\n")
            for line in lines_to_add:
                f.write(f"{line}\n")
        print(f"Added environment variables to {config_path}")
        print(f"  Run 'source {config_path}' or restart your shell to use them.")
    else:
        print(f"Environment variables already configured in {config_path}")


def download_version(version: str, output_dir: Path) -> None:
    """Download a specific version of the dataset."""
    if version == "v1":
        revision = V1_BRANCH
    elif version == "v2":
        revision = V2_TAG
    else:
        raise ValueError(f"Unknown version: {version}")
    
    print(f"\nDownloading")
    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    

def main():
    parser = argparse.ArgumentParser(
        description="Download PILArNet-M dataset from Hugging Face"
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "both"],
        default="both",
        help="Which version to download (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base output directory (default: ~/.cache/tpcfm/pilarnet)",
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        help="Output directory for v1 (default: <output-dir>/v1)",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        help="Output directory for v2 (default: <output-dir>/v2)",
    )
    parser.add_argument(
        "--no-env-setup",
        action="store_true",
        help="Skip asking about environment variable setup",
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path.home() / ".cache" / "pimm" / "pilarnet"
    
    if args.v1_dir is None:
        args.v1_dir = args.output_dir / "v1"
    if args.v2_dir is None:
        args.v2_dir = args.output_dir / "v2"
    
    # Download requested versions
    if args.version in ["v1", "both"]:
        download_version("v1", args.v1_dir)
    
    if args.version in ["v2", "both"]:
        download_version("v2", args.v2_dir)
    
    # Ask about environment variables
    if not args.no_env_setup:
        print("\n" + "=" * 60)
        response = input(
            "Would you like to add PILARNET_DATA_ROOT_V1 and PILARNET_DATA_ROOT_V2 "
            "to your shell config file? [y/N]: "
        ).strip().lower()
        
        if response in ["y", "yes"]:
            add_to_shell_config(args.v1_dir, args.v2_dir)
        else:
            print("Skipping environment variable setup.")
            print("You can manually set:")
            print(f'  export PILARNET_DATA_ROOT_V1="{args.v1_dir}"')
            print(f'  export PILARNET_DATA_ROOT_V2="{args.v2_dir}"')
    
    print("\nDownload complete")
    print("\nDataset locations:")
    if args.version in ["v1", "both"]:
        print(f"  v1: {args.v1_dir}")
    if args.version in ["v2", "both"]:
        print(f"  v2: {args.v2_dir}")


if __name__ == "__main__":
    main()

