"""List the datasets pimm reads, and download one from the Hugging Face Hub."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Literal

import tyro
from huggingface_hub import snapshot_download

from pimm.datasets.catalog import CACHE, CATALOG, data_home

Name = Literal[tuple(CATALOG)]


def list_datasets() -> None:
    """Print every dataset in the catalog."""
    rows = [("name", "repo", "license", "size", "splits", "revisions")]
    for name, d in CATALOG.items():
        rows.append((name, d["repo"], d["license"], d["size"], " ".join(d["splits"]) or "-", " ".join(d.get("revisions", ())) or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def ask(question: str) -> bool:
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


def shell_config() -> Path:
    shell, home = os.environ.get("SHELL", ""), Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    return home / ".bash_profile" if (home / ".bash_profile").exists() else home / ".bashrc"


def download(
    name: tyro.conf.Positional[Name],
    split: tuple[str, ...] = (),
    revision: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Download a dataset, or some of its splits, to OUTPUT_DIR, which is $PIMM_DATA_ROOT/NAME or else
    ~/.cache/pimm/datasets/NAME unless you set it.

    Args:
        name: A dataset from `pimm datasets list`.
        split: Splits to download; leave it out for every split.
        revision: The revision, for datasets that have several; leave it out for the first one listed.
        output_dir: Where the files go; leave it out for the default above.
    """
    entry = CATALOG[name]
    revisions = entry.get("revisions")
    revision = revision or (next(iter(revisions)) if revisions else None)
    sub = f"{name}/{revision}" if revisions else name
    dest = (output_dir or data_home(sub)).absolute()
    workers = int(input("Download workers [8]: ").strip() or 8)
    patterns = [p for s in split for p in entry["splits"][s]] or None
    print(f"Downloading {entry['repo']} ({', '.join(split) or 'all splits'}) to {dest}")
    snapshot_download(
        entry["repo"],
        repo_type="dataset",
        revision=(revisions or {}).get(revision, revision),
        allow_patterns=patterns,
        local_dir=dest,
        max_workers=workers,
    )

    if name == "next-aiss":
        tars = sorted(dest.glob("*.tar"))
        if ask(f"Extract the {len(tars)} tar parts into {dest} now (the .tar files are kept)?"):
            for path in tars:
                print(f"Extracting {path.name}")
                with tarfile.open(path) as archive:
                    archive.extractall(dest, filter="data")
        else:
            print(f"To extract later: for f in {shlex.quote(str(dest))}/*.tar; do tar -xf \"$f\" -C {shlex.quote(str(dest))}; done")
    if name == "tpcpp-10m" and (not split or "unlabeled" in split):
        command = [sys.executable, "-m", "pimm.datasets.preprocessing.tpcpp10m", str(dest), "--workers", str(workers)]
        if ask("Write the unlabeled shards uncompressed now (about 139 GB more)?"):
            subprocess.run(command, check=True)
        else:
            print(f"To write them later: {shlex.join(['python'] + command[1:])}")

    # The lookup finds `dest` through PIMM_DATA_ROOT only when `dest` is <root>/<sub>.
    root = str(dest).removesuffix(f"/{sub}")
    if "PIMM_DATA_ROOT" not in os.environ and root not in (str(dest), str(CACHE)):
        line = f'export PIMM_DATA_ROOT="{root}"'
        config = shell_config()
        if ask(f"Add `{line}` to {config}?"):
            with config.open("a") as f:
                f.write(f"\n# pimm datasets\n{line}\n")
            print(f"Run `source {config}` or open a new shell to use it.")
        else:
            print(f"To set it yourself: {line}")
    print(f"{name} is in {dest}")


def main(argv: list[str] | None = None) -> int:
    tyro.extras.subcommand_cli_from_dict(
        {"list": list_datasets, "download": download}, args=argv, prog="pimm datasets"
    )
    return 0
