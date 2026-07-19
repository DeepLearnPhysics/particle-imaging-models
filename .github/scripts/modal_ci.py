"""Run the pimm test suite on Modal GPUs for trusted branch commits.

GitHub-hosted runners have no GPU, so this app ships the committed checkout to
Modal, builds the locked wheel-based environment (no native compile), primes a
shared volume with the commit-pinned PILArNet-M-mini dataset and released
models, and runs the shared pytest suite on CPU plus A100, L40S, and H100 workers.

Run locally with:

    uvx modal run .github/scripts/modal_ci.py

Set PIMM_IMAGE to validate a published application image instead of the
checkout; the suite then runs in the image's venv with the checkout mounted,
mirroring how containerized jobs use the image:

    PIMM_IMAGE=youngsm/pimm:pytorch2.10.0-cuda12.6 uvx modal run .github/scripts/modal_ci.py
"""

import json
import os
import subprocess
from pathlib import Path

import modal


# remote containers re-import this module from a shallower path; PIMM_IMAGE
# reaches them through the baked image env, so both sides resolve one layout
_here = Path(__file__).resolve()
REPO_ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
PIMM_IMAGE = os.environ.get("PIMM_IMAGE", "")
if PIMM_IMAGE:
    # the published image ships only the venv; the checkout is mounted at the
    # image's default bind point, mirroring how jobs run it
    REMOTE_ROOT = "/opt/pimm/src"
    VENV = "/opt/pimm/.venv"
else:
    REMOTE_ROOT = "/opt/pimm"
    # venv lives outside the copied source dir: writes under an add_local_dir
    # target during a build function do not persist into the image layer
    VENV = "/opt/pimm-venv"
DATA_ROOT = "/opt/pimm-data/pilarnet"
HF_CACHE = "/opt/pimm-data/hf-cache"
MANIFEST = "tests/integration/fixtures/metric_baselines.json"

UV_VERSION = "0.11.28"
INTEGRATION_TIMEOUT = 1800
OUTPUT_TAIL = 8000

RUNTIME_ENV = {
    "HF_HUB_DISABLE_XET": "1",
    "PILARNET_DATA_ROOT_V2": DATA_ROOT,
    "HF_HUB_CACHE": HF_CACHE,
    "PIMM_IMAGE": PIMM_IMAGE,
    "PIMM_TEST_DATA_ROOT": DATA_ROOT,
    "PYTHONPATH": REMOTE_ROOT,
    "UV_PROJECT_ENVIRONMENT": VENV,
    "WANDB_MODE": "disabled",
}


def _sync_environment():
    """Install the locked environment (plus the dev group) into the image.

    A build function rather than a run_commands layer: uv's ``--locked`` check
    passes in the function sandbox but rejects the lock in a build layer.
    In image mode this re-verifies the published image's venv against its
    lockfile and adds pytest on top.
    """
    command = ["uv", "sync", "--locked", "--group", "dev"]
    if PIMM_IMAGE:
        # keep the published venv project-free; the suite imports the mounted
        # checkout through PYTHONPATH, matching how jobs use the image
        command.append("--no-install-project")
    subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": VENV,
            "UV_CACHE_DIR": "/tmp/uv-cache",
        },
        check=True,
    )


data_volume = modal.Volume.from_name("pimm-ci-data", create_if_missing=True)
VOLUMES = {"/opt/pimm-data": data_volume}


if PIMM_IMAGE:
    # validate the published application image: run the suite against its baked
    # source and venv. The env layer comes first so _sync_environment (which
    # re-imports this module) sees PIMM_IMAGE and resolves the image-mode paths.
    image = (
        modal.Image.from_registry(PIMM_IMAGE, add_python="3.10")
        # not pip_install: the image's PATH resolves `python` to its pip-less
        # uv venv, so target Modal's interpreter explicitly
        .run_commands(
            f"/usr/local/bin/python -m pip install huggingface_hub uv=={UV_VERSION}"
        )
        .add_local_dir(
            REPO_ROOT,
            REMOTE_ROOT,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                ".venv*",
                "**/__pycache__",
                "**/*.pyc",
                "**/build",
                "**/*.egg-info",
                "**/*.so",
                "docs/build",
            ],
        )
        .env(RUNTIME_ENV)
        .run_function(_sync_environment)
    )
else:
    image = (
        # base CUDA/cuDNN must match the locked torch stack (torch 2.10.0, cu126)
        modal.Image.from_registry(
            "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel",
            add_python="3.10",
        )
        .pip_install("huggingface_hub", f"uv=={UV_VERSION}")
        .add_local_dir(
            REPO_ROOT,
            REMOTE_ROOT,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                ".venv*",
                "**/__pycache__",
                "**/*.pyc",
                "**/build",
                "**/*.egg-info",
                "**/*.so",
                "docs/build",
            ],
        )
        .run_function(_sync_environment)
        .env(RUNTIME_ENV)
    )

app = modal.App("pimm-ci", image=image)


@app.function(volumes=VOLUMES, timeout=INTEGRATION_TIMEOUT)
def prime_cache():
    """Populate the shared dataset/model volume, verifying every hash.

    A regular function on a volume rather than an image build step: baking
    the data into an image layer proved unreliable (a mid-build container
    restart committed a layer without the downloaded files), and the volume
    survives image rebuilds so the data downloads only once.
    """
    import hashlib

    from huggingface_hub import hf_hub_download, snapshot_download

    manifest = json.loads(Path(REMOTE_ROOT, MANIFEST).read_text())
    dataset = manifest["dataset"]

    for filename, info in dataset["files"].items():
        path = Path(DATA_ROOT, filename)
        if not path.is_file():
            hf_hub_download(
                repo_id=dataset["repo_id"],
                repo_type=dataset["repo_type"],
                revision=dataset["revision"],
                filename=filename,
                local_dir=DATA_ROOT,
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise SystemExit(f"sha256 mismatch for {filename}: {digest}")

    for baseline in manifest["baselines"].values():
        repo_id = baseline["model_uri"].removeprefix("hf://").split("@", 1)[0]
        snapshot_download(
            repo_id=repo_id,
            revision=baseline["model_revision"],
            cache_dir=HF_CACHE,
        )
    data_volume.commit()


def _pytest(*args):
    """Run pytest with the given args in the locked environment and report the result."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "pytest", "-v", *args],
        cwd=REMOTE_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output)
    return {"returncode": result.returncode, "tail": output[-OUTPUT_TAIL:]}


# single-GPU cells skip the distributed tests; the A10:2 cell runs only those
SINGLE_GPU_ARGS = ("tests/integration", "-m", "not distributed")


@app.function()
def unit():
    return _pytest("tests/unit")


@app.function(gpu="A100", timeout=INTEGRATION_TIMEOUT, volumes=VOLUMES)
def integration_a100():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="L40S", timeout=INTEGRATION_TIMEOUT, volumes=VOLUMES)
def integration_l40s():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="H100", timeout=INTEGRATION_TIMEOUT, volumes=VOLUMES)
def integration_h100():
    return _pytest(*SINGLE_GPU_ARGS)


@app.function(gpu="A10:2", timeout=INTEGRATION_TIMEOUT, volumes=VOLUMES)
def integration_distributed():
    return _pytest("tests/integration", "-m", "distributed")


@app.local_entrypoint()
def main():
    # the shared data volume must be primed before any integration cell starts
    prime_cache.remote()

    # image mode certifies one published artifact; the full GPU matrix already
    # ran against the same commit through the checkout-based suite
    calls = {
        "cpu-unit": unit.spawn(),
        "a100-integration": integration_a100.spawn(),
    }
    if not PIMM_IMAGE:
        calls.update(
            {
                "l40s-integration": integration_l40s.spawn(),
                "h100-integration": integration_h100.spawn(),
                "a10x2-distributed": integration_distributed.spawn(),
            }
        )
    results = {name: call.get() for name, call in calls.items()}

    failed = []
    for name, result in results.items():
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        print(f"[{status}] {name} (returncode={result['returncode']})")
        if result["returncode"] != 0:
            print(result["tail"])
            failed.append(name)

    target = PIMM_IMAGE or "the checkout"
    if failed:
        raise SystemExit(f"modal ci failed for {target}: " + ", ".join(sorted(failed)))
    print(f"modal ci passed for {target} on: " + ", ".join(sorted(calls)))
