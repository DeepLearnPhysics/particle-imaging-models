"""Build and validate one native compatibility stack.

This script runs inside the manylinux builder selected by native-wheels.json.
It is shared by GitHub Actions and local release preparation so both paths
produce the same wheel set and manifest.
"""

import email
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(os.environ.get("PIMM_ROOT", "/workspace")).resolve()
OUT = Path(os.environ.get("WHEELHOUSE", ROOT / "wheelhouse")).resolve()
CONFIG = ROOT / ".github/scripts/native-wheels.json"


def canonical(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def verify_source(source: dict, package: dict) -> None:
    sha256 = source["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise SystemExit(f"invalid source hash for {package['distribution']}")
    if not source["url"].startswith("https://github.com/"):
        raise SystemExit(
            f"source URL must use GitHub HTTPS for {package['distribution']}"
        )


def extract_archive(archive: Path, destination: Path) -> None:
    unpacked = archive.with_suffix("")
    unpacked.mkdir()
    with tarfile.open(archive) as handle:
        root = unpacked.resolve()
        for member in handle.getmembers():
            target = (unpacked / member.name).resolve()
            target.relative_to(root)
            if member.issym() or member.islnk():
                link = (target.parent / member.linkname).resolve()
                link.relative_to(root)
        handle.extractall(unpacked)
    entries = list(unpacked.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise SystemExit(f"archive must contain one source root: {archive.name}")
    shutil.copytree(entries[0], destination)


def download_source(source: dict, destination: Path, package: dict) -> None:
    verify_source(source, package)
    if destination.exists():
        shutil.rmtree(destination)
    archive = destination.parent / f"{destination.name}.tar.gz"
    digest = hashlib.sha256()
    with urllib.request.urlopen(source["url"], timeout=60) as response, archive.open(
        "wb"
    ) as out:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != source["sha256"]:
        raise SystemExit(f"source hash mismatch for {package['distribution']}")
    extract_archive(archive, destination)


def build_source(package: dict, source_root: Path) -> str:
    if "path" in package:
        source = (ROOT / package["path"]).resolve()
        source.relative_to(ROOT / "libs")
        return str(source)

    source = package["source"]
    destination = source_root / canonical(package["distribution"])
    download_source(source, destination, package)
    for submodule in source.get("submodules", []):
        relative = Path(submodule["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"invalid submodule path for {package['distribution']}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        download_source(submodule, target, package)
    return str(destination)


def select_packages(stack: dict) -> list[dict]:
    requested = {
        canonical(item)
        for item in os.environ.get("PIMM_PACKAGE_FILTER", "").split(",")
        if item
    }
    if not requested:
        return stack["packages"]
    packages = [
        package
        for package in stack["packages"]
        if canonical(package["distribution"]) in requested
    ]
    found = {canonical(package["distribution"]) for package in packages}
    if found != requested:
        raise SystemExit(f"unknown package filter: {sorted(requested - found)}")
    return packages


def package_version(package: dict, stack: dict) -> str:
    if "version" in package:
        return package["version"]
    torch_minor = ".".join(stack["torch"].split("+", 1)[0].split(".")[:2])
    return f"{package['base_version']}+{stack['cuda_tag']}torch{torch_minor}"


def audit_glibc(wheel: Path) -> None:
    """Reject binaries that cannot load on the RHEL 8 cluster baseline."""
    with tempfile.TemporaryDirectory(prefix="pimm-wheel-audit-") as tmp:
        with zipfile.ZipFile(wheel) as archive:
            binaries = [name for name in archive.namelist() if name.endswith(".so")]
            for member in binaries:
                archive.extract(member, tmp)
                dump = subprocess.run(
                    ["objdump", "-T", str(Path(tmp, member))],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                ceiling = max(
                    (
                        tuple(map(int, match.group(1).split(".")))
                        for match in re.finditer(r"GLIBC_([0-9.]+)", dump)
                    ),
                    default=(0,),
                )
                printable = ".".join(map(str, ceiling))
                print(f"{wheel.name}:{member} GLIBC ceiling {printable}")
                if ceiling > (2, 28):
                    raise SystemExit(
                        f"glibc ceiling above 2.28 in {wheel.name}: {member}"
                    )


def audit_cuda_arches(wheel: Path, arches: list[str]) -> None:
    """Require device code for every GPU architecture in the stack."""
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cuobjdump = shutil.which("cuobjdump") or str(cuda_home / "bin/cuobjdump")
    expected = {arch.replace(".", "") for arch in arches}
    found = set()
    with tempfile.TemporaryDirectory(prefix="pimm-wheel-cuda-audit-") as tmp:
        with zipfile.ZipFile(wheel) as archive:
            binaries = [name for name in archive.namelist() if name.endswith(".so")]
            for member in binaries:
                archive.extract(member, tmp)
                result = subprocess.run(
                    [cuobjdump, "--list-elf", str(Path(tmp, member))],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                found.update(
                    re.findall(r"\bsm_([0-9]+)\b", result.stdout + result.stderr)
                )
    missing = expected - found
    printable = ", ".join(f"sm_{arch}" for arch in sorted(found))
    print(f"{wheel.name} CUDA architectures: {printable}")
    if missing:
        missing_text = ", ".join(f"sm_{arch}" for arch in sorted(missing))
        raise SystemExit(f"missing CUDA architectures in {wheel.name}: {missing_text}")


def inspect_wheel(
    wheel: Path, package: dict, version: str, tag: str, arches: list[str]
) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise SystemExit(f"invalid metadata in {wheel.name}")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        wheel_metadata = email.message_from_bytes(archive.read(wheel_names[0]))

    if canonical(metadata["Name"]) != canonical(package["distribution"]):
        raise SystemExit(f"unexpected distribution: {wheel.name}")
    if metadata["Version"] != version:
        raise SystemExit(f"unexpected version: {wheel.name}")
    if tag not in wheel_metadata.get_all("Tag", []):
        raise SystemExit(f"unexpected wheel tag: {wheel.name}")

    audit_glibc(wheel)
    audit_cuda_arches(wheel, arches)
    return {
        "filename": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "distribution": package["distribution"],
        "version": version,
        "import_name": package["import_name"],
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    stack_id = os.environ["STACK"]
    matches = [item for item in config["stacks"] if item["id"] == stack_id]
    if len(matches) != 1:
        raise SystemExit(f"unknown or duplicate compatibility stack: {stack_id}")
    stack = matches[0]

    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python != stack["python"]:
        raise SystemExit(f"expected Python {stack['python']}, got {python}")

    import torch

    if torch.__version__ != stack["torch"] or torch.version.cuda != stack["cuda"]:
        raise SystemExit(
            f"unexpected torch stack: {torch.__version__}, CUDA {torch.version.cuda}"
        )

    packages = select_packages(stack)
    source_root = Path(
        os.environ.get("PIMM_SOURCE_ROOT", "/tmp/pimm-native-sources")
    ).resolve()
    shutil.rmtree(source_root, ignore_errors=True)
    source_root.mkdir(parents=True)

    OUT.mkdir(parents=True, exist_ok=True)
    reuse_wheels = os.environ.get("PIMM_REUSE_WHEELS") == "1"
    if not reuse_wheels:
        for stale in OUT.glob("*.whl"):
            stale.unlink()
    # NVCC gives intermediate host objects process-specific ``tmpxft`` symbol
    # names. Release wheels do not need local/debug symbols, so strip them at
    # link time to reduce build-specific metadata and make the payload smaller.
    build_env = os.environ.copy()
    strip_flag = "-Wl,--strip-all"
    linker_flags = build_env.get("LDFLAGS", "").split()
    if strip_flag not in linker_flags:
        linker_flags.append(strip_flag)
    build_env["LDFLAGS"] = " ".join(linker_flags)
    if not reuse_wheels:
        for package in packages:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-deps",
                    "--wheel-dir",
                    str(OUT),
                    build_source(package, source_root),
                ],
                check=True,
                env=build_env,
            )

    wheels = sorted(OUT.glob("*.whl"))
    if len(wheels) != len(packages):
        raise SystemExit(
            f"expected {len(packages)} wheels, found {len(wheels)}"
        )

    tag = f"{stack['python_tag']}-{stack['python_tag']}-linux_x86_64"
    records = []
    for package in packages:
        version = package_version(package, stack)
        name = re.sub(r"[-_.]+", "_", package["distribution"])
        wheel = OUT / f"{name}-{version}-{tag}.whl"
        if not wheel.is_file():
            raise SystemExit(f"missing expected wheel: {wheel.name}")
        records.append(inspect_wheel(wheel, package, version, tag, stack["arches"]))

    # Install the exact artifacts, not their source trees, before the import
    # check. This catches missing shared objects and Torch ABI mismatches.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-index",
            "--no-deps",
            *[str(OUT / record["filename"]) for record in records],
        ],
        check=True,
    )
    for package, record in zip(packages, records):
        if "validation_library" in package:
            distribution = importlib.metadata.distribution(record["distribution"])
            library = distribution.locate_file(package["validation_library"])
            torch.ops.load_library(str(library))
        else:
            importlib.import_module(record["import_name"])

    (OUT / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in records)
    )
    release = {
        "schema_version": 1,
        "repository": config["repository"],
        "release_tag": stack["release_tag"],
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "stack": {
            key: stack[key]
            for key in ("id", "python", "python_tag", "torch", "cuda", "arches")
        },
        "wheels": records,
    }
    (OUT / "native-wheels-manifest.json").write_text(
        json.dumps(release, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
