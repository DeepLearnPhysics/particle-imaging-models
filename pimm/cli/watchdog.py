#!/usr/bin/env python3
"""Manage watchdogs that supervise chained interactive runs.

A chained `pimm submit --interactive` run installs a scron-hosted watchdog (see
pimm/launch/watchdog.py). This command lists and removes them. Installation
happens through `pimm submit`, not here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pimm.launch import watchdog


def _ls() -> int:
    entries = watchdog.parsed_entries()
    if not entries:
        print("no watchdogs installed")
        return 0
    width = max(len(e["run"]) for e in entries)
    for e in entries:
        st = watchdog.dir_state(e.get("state_dir", ""), e.get("job", e["run"]))
        print(f"  {e['run']:<{width}}  {st:<8}  {e.get('state_dir', '')}")
    return 0


def _rm(run: str, scancel: bool) -> int:
    entry = next((e for e in watchdog.parsed_entries() if e["run"] == run), None)
    if not watchdog.remove_entry(run):
        print(f"no watchdog installed for run: {run}")
        return 1
    print(f"removed watchdog scron entry for run: {run}")
    if scancel and entry:
        job = entry.get("job", run)
        for name in (f"{job}-watchdog", job):
            subprocess.run(["scancel", "-n", name], capture_output=True, text=True)
        print(f"cancelled running driver + slot for: {run}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pimm watchdog")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("ls", help="list installed watchdogs and their state")
    rm = sub.add_parser("rm", help="remove a run's watchdog (stop supervising)")
    rm.add_argument("run", help="run name (as shown by `pimm watchdog ls`)")
    rm.add_argument(
        "--scancel",
        action="store_true",
        help="also scancel the running driver and its interactive slot",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.action == "ls":
        return _ls()
    return _rm(args.run, args.scancel)


if __name__ == "__main__":
    raise SystemExit(main())
