"""Verify that uv.lock pins the exact native release artifacts."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 used by pimm
    import tomli as tomllib


def canonical(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    lock = tomllib.loads(args.lock.read_text())
    project = tomllib.loads(args.project.read_text())
    packages = {}
    for package in lock["package"]:
        packages.setdefault(canonical(package["name"]), []).append(package)

    release_tag = manifest["release_tag"]
    records = {
        canonical(record["distribution"]): record for record in manifest["wheels"]
    }
    if len(records) != 9 or len(records) != len(manifest["wheels"]):
        raise SystemExit("native wheel manifest must contain nine unique distributions")

    expected_files = {
        record["filename"] for record in manifest["wheels"]
    } | {args.manifest.name, "SHA256SUMS"}
    actual_files = {
        path.name for path in args.manifest.parent.iterdir() if path.is_file()
    }
    if actual_files != expected_files:
        raise SystemExit(
            "wheelhouse contents do not match the manifest: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    project_sources = {
        canonical(name): source
        for name, source in project["tool"]["uv"]["sources"].items()
        if isinstance(source, dict)
        and f"/releases/download/{release_tag}/" in source.get("url", "")
    }
    if set(project_sources) != set(records):
        raise SystemExit("pyproject native wheel sources do not match the manifest")

    for record in manifest["wheels"]:
        distribution = canonical(record["distribution"])
        artifact = args.manifest.parent / record["filename"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise SystemExit(f"artifact hash mismatch for {record['filename']}")

        project_url = project_sources[distribution]["url"]
        if unquote(Path(urlparse(project_url).path).name) != record["filename"]:
            raise SystemExit(
                f"pyproject source mismatch for {record['distribution']}"
            )

        matches = packages.get(distribution, [])
        matches = [item for item in matches if item["version"] == record["version"]]
        if len(matches) != 1:
            raise SystemExit(
                f"uv.lock must contain one {record['distribution']}=={record['version']}"
            )
        package = matches[0]
        source_url = package.get("source", {}).get("url", "")
        if f"/releases/download/{release_tag}/" not in source_url:
            raise SystemExit(
                f"{record['distribution']} does not use release {release_tag}"
            )
        if unquote(Path(urlparse(source_url).path).name) != record["filename"]:
            raise SystemExit(f"unexpected source filename for {record['distribution']}")

        wheels = [
            wheel
            for wheel in package.get("wheels", [])
            if unquote(Path(urlparse(wheel["url"]).path).name) == record["filename"]
        ]
        expected_hash = f"sha256:{record['sha256']}"
        if len(wheels) != 1 or wheels[0].get("hash") != expected_hash:
            raise SystemExit(
                f"uv.lock hash mismatch for {record['filename']}: expected {expected_hash}"
            )

    print(
        f"validated {len(manifest['wheels'])} project and lock wheel pins "
        f"for {release_tag}"
    )


if __name__ == "__main__":
    main()
