"""Regenerate checked-in tutorial figures on one Modal L4 GPU.

The interactive documentation keeps its plotting code in Jupytext-compatible
Python notebooks.  This app runs those same sources on a GPU and returns the
generated HTML, PNG, and metadata files to the caller for review.

Run from a trusted checkout with Modal credentials in the environment:

    uvx modal run .github/scripts/modal_docs_figures.py

No generated file is committed or published automatically.
"""

from __future__ import annotations

import io
import os
import subprocess
import zipfile
from pathlib import Path

import modal


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent
REMOTE_ROOT = "/opt/pimm"
VENV = "/opt/pimm-venv"
REMOTE_OUTPUT = "/tmp/pimm-doc-figures"
LOCAL_OUTPUT = REPO_ROOT / "docs" / "source" / "_static" / "tutorials"
UV_VERSION = "0.11.28"

RUNTIME_ENV = {
    "HF_HOME": "/cache/huggingface",
    "HF_HUB_CACHE": "/cache/huggingface/hub",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "PYTHONPATH": REMOTE_ROOT,
    "UV_PROJECT_ENVIRONMENT": VENV,
    "WANDB_MODE": "disabled",
}


def _sync_environment():
    """Install the locked pimm environment into the image."""
    subprocess.run(
        ["uv", "sync", "--locked", "--group", "dev"],
        cwd=REMOTE_ROOT,
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": VENV,
            "UV_CACHE_DIR": "/tmp/uv-cache",
        },
        check=True,
    )


image = (
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
            "docs/source/_static/tutorials/*.html",
            "docs/source/_static/tutorials/*.png",
            "docs/source/_static/tutorials/*.json",
            "docs/source/_static/tutorials/plotly.min.js",
        ],
    )
    .run_function(_sync_environment)
    .env(RUNTIME_ENV)
)

cache = modal.Volume.from_name("pimm-doc-figure-cache", create_if_missing=True)
app = modal.App("pimm-doc-figures", image=image)


def _run_notebook(source: str, *, event: int, seed: int):
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "--with",
            "plotly",
            "python",
            source,
            "--models",
            "all",
            "--device",
            "cuda",
            "--event",
            str(event),
            "--seed",
            str(seed),
            "--output-dir",
            REMOTE_OUTPUT,
        ],
        cwd=REMOTE_ROOT,
        check=True,
    )


@app.function(gpu="L4", timeout=3600, volumes={"/cache": cache})
def generate_figures(models: str = "all", event: int = 0, seed: int = 7) -> bytes:
    """Run the selected notebook sources and return one ZIP archive."""
    output = Path(REMOTE_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)

    if models in {"all", "panda"}:
        _run_notebook(
            "docs/source/tutorials/explore_panda.py", event=event, seed=seed
        )
    if models in {"all", "polarmae"}:
        _run_notebook(
            "docs/source/tutorials/explore_polarmae.py", event=event, seed=seed
        )

    cache.commit()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    return payload.getvalue()


def _extract_generated(payload: bytes, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.name != info.filename or path.suffix not in {".html", ".png", ".json", ".js"}:
                raise RuntimeError(f"Unexpected generated archive member: {info.filename}")
        archive.extractall(destination)


@app.local_entrypoint()
def main(models: str = "all", event: int = 0, seed: int = 7):
    """Generate figures remotely and replace the matching local assets."""
    if models not in {"all", "panda", "polarmae"}:
        raise ValueError("models must be one of: all, panda, polarmae")
    payload = generate_figures.remote(models=models, event=event, seed=seed)
    _extract_generated(payload, LOCAL_OUTPUT)
    print(f"Wrote reviewed figure candidates to {LOCAL_OUTPUT}")
