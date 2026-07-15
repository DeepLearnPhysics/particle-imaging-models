"""Regenerate the interactive figures used by the Explore Panda documentation.

This is a Jupytext-compatible ``percent`` notebook: open it as a normal Python
file in an editor that understands ``# %%`` cells, convert it to ``.ipynb``
with Jupytext, or run it from beginning to end as a script.

The dataset-only path runs on CPU.  ``semantic``, ``base``, ``particle``, and
``interaction`` load released Panda checkpoints and currently require CUDA
because their PTv3 backbone uses spconv.
"""

# %% [markdown]
# # Explore Panda, reproducibly
#
# This notebook downloads one deterministic event from PILArNet-M-mini,
# applies the same test-time transform contract used by the released Panda
# models, and writes responsive Plotly figures.  In a notebook, edit the model
# selection in the final cell; when running this file as a script, pass model
# names to ``--models``.  Released model inference currently needs CUDA.

# %%
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MOTIF_NAMES = ("Shower", "Track", "Michel", "Delta", "Low-energy deposit")
MOTIF_COLORS = ("#5b8ff9", "#f6bd16", "#5ad8a6", "#e8684a", "#9270ca")
PID_NAMES = ("Photon", "Electron", "Muon", "Pion", "Proton", "None / LED")
PID_COLORS = ("#5b8ff9", "#61dDAa", "#f6bd16", "#7262fd", "#e8684a", "#8c8c8c")
INSTANCE_COLORS = (
    "#5b8ff9", "#5ad8a6", "#5d7092", "#f6bd16", "#e8684a", "#6dc8ec",
    "#9270ca", "#ff9d4d", "#269a99", "#ff99c3", "#6f5ef9", "#f08bb4",
    "#65789b", "#f6bd16", "#9661bc", "#5ad8a6", "#e86452", "#6dc8ec",
    "#ff9845", "#1e9493",
)

MODEL_REPOS = {
    "base": "DeepLearnPhysics/Panda-Base",
    "semantic": "DeepLearnPhysics/Panda-Semantic",
    "particle": "DeepLearnPhysics/Panda-Particle",
    "interaction": "DeepLearnPhysics/Panda-Interaction",
}

# Deliberately no random augmentation: these are the deterministic inference
# transforms from the Panda semantic fine-tuning recipe.
TEST_TRANSFORM = [
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "energy",
            "segment_motif",
            "segment_pid",
            "instance_particle",
            "instance_interaction",
            "segment_interaction",
        ),
        feat_keys=("coord", "energy"),
    ),
]


# %% [markdown]
# ## Plotting helpers
#
# The HTML figures all share one local Plotly runtime, so the rendered docs work
# without a CDN and the browser only downloads the library once.

# %%
def _as_numpy(value):
    """Detach a tensor if needed and return a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _flatten(value) -> np.ndarray:
    return _as_numpy(value).reshape(-1)


def _model_input(sample: Mapping, device: str):
    """Keep only the public Panda input contract and move it to ``device``."""
    import torch

    result = {}
    for key in ("coord", "grid_coord", "feat", "offset"):
        value = sample[key]
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _configure_scene(fig, *, title: str):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
    )
    scene_updates = dict(
        aspectmode="data",
        xaxis=dict(title="x", showbackground=False),
        yaxis=dict(title="y", showbackground=False),
        zaxis=dict(title="z", showbackground=False),
        camera=dict(eye=dict(x=1.45, y=1.45, z=1.1)),
    )
    fig.update_scenes(**scene_updates)
    return fig


def _class_traces(
    coords: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    colors: Sequence[str],
    *,
    scene: str = "scene",
    legendgroup: str,
    showlegend: bool = True,
    hover_suffix: Sequence[str] | None = None,
):
    import plotly.graph_objects as go

    traces = []
    for class_id in np.unique(labels):
        class_id = int(class_id)
        mask = labels == class_id
        if not np.any(mask):
            continue
        known = 0 <= class_id < len(names)
        name = names[class_id] if known else f"Label {class_id}"
        color = colors[class_id % len(colors)]
        suffix = "" if hover_suffix is None else f"<br>{hover_suffix[class_id]}"
        traces.append(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                marker=dict(size=2.7, color=color, opacity=0.88),
                name=f"{name} ({int(mask.sum()):,})",
                legendgroup=f"{legendgroup}-{class_id}",
                showlegend=showlegend,
                scene=scene,
                hovertemplate=(
                    f"{name}{suffix}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}"
                    "<br>z=%{z:.4f}<extra></extra>"
                ),
            )
        )
    return traces


def _comparison_figure(
    coords: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    left_title: str,
    right_title: str,
    names: Sequence[str],
    colors: Sequence[str],
    right_names: Sequence[str] | None = None,
    right_colors: Sequence[str] | None = None,
    title: str,
):
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(left_title, right_title),
        horizontal_spacing=0.02,
    )
    for trace in _class_traces(
        coords, left_labels, names, colors, legendgroup="left", showlegend=True
    ):
        fig.add_trace(trace, row=1, col=1)
    # Keep the right legend too: the two panels can contain different classes.
    for trace in _class_traces(
        coords,
        right_labels,
        right_names or names,
        right_colors or colors,
        legendgroup="right",
        showlegend=True,
    ):
        fig.add_trace(trace, row=1, col=2)
    return _configure_scene(fig, title=title)


def _instance_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
):
    import plotly.graph_objects as go

    fig = go.Figure()
    for index, instance_id in enumerate(np.unique(labels)):
        instance_id = int(instance_id)
        mask = labels == instance_id
        label = "Unassigned" if instance_id < 0 else f"Instance {instance_id}"
        color = "#aaaaaa" if instance_id < 0 else INSTANCE_COLORS[index % len(INSTANCE_COLORS)]
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                marker=dict(size=2.7, color=color, opacity=0.88),
                name=f"{label} ({int(mask.sum()):,})",
                hovertemplate=(
                    f"{label}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}"
                    "<br>z=%{z:.4f}<extra></extra>"
                ),
            )
        )
    return _configure_scene(fig, title=title)


def _energy_figure(coords: np.ndarray, energy: np.ndarray, *, title: str):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=2.8,
                color=energy,
                colorscale="Viridis",
                colorbar=dict(title="log-scaled energy"),
                opacity=0.9,
            ),
            hovertemplate=(
                "energy=%{marker.color:.4f}<br>x=%{x:.4f}<br>y=%{y:.4f}"
                "<br>z=%{z:.4f}<extra></extra>"
            ),
        )
    )
    return _configure_scene(fig, title=title)


def _feature_figure(coords: np.ndarray, rgb: np.ndarray, *, title: str):
    import plotly.graph_objects as go

    rgb255 = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb255]
    fig = go.Figure(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=2.8, color=colors, opacity=0.9),
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )
    return _configure_scene(fig, title=title)


def _write_figure(fig, output_dir: Path, stem: str):
    """Write responsive HTML and share one local Plotly runtime across figures."""
    output = output_dir / f"{stem}.html"
    fig.write_html(
        output,
        include_plotlyjs="directory",
        full_html=True,
        div_id=stem,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    return output


def _write_static(
    coords: np.ndarray,
    panels: Sequence[tuple[str, np.ndarray]],
    output: Path,
    *,
    categorical: bool = True,
):
    """Generate a no-JavaScript fallback without requiring Plotly/Kaleido."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2 * len(panels), 5.4))
    for column, (title, values) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), column, projection="3d")
        if values.ndim == 2 and values.shape[1] == 3:
            color = np.clip(values, 0, 1)
            cmap = None
        else:
            color = values.reshape(-1)
            cmap = "tab20" if categorical else "viridis"
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, cmap=cmap, s=1.2)
        ax.set(title=title, xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect(np.ptp(coords, axis=0))
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# %% [markdown]
# ## Dataset views
#
# These views need no model or GPU.  Alongside the interactive HTML, this cell
# writes static PNG fallbacks and a small JSON file with the exact point counts.

# %%
def render_dataset_figures(
    sample: Mapping,
    output_dir: Path,
    *,
    event: int,
    show: bool = False,
):
    """Render the CPU-only figures and return reproducibility metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = _as_numpy(sample["coord"])
    energy = _flatten(sample["energy"])
    motif = _flatten(sample["segment_motif"]).astype(int)
    pid = _flatten(sample["segment_pid"]).astype(int)
    particle = _flatten(sample["instance_particle"]).astype(int)
    interaction = _flatten(sample["instance_interaction"]).astype(int)

    figures = {
        "panda-event-energy": _energy_figure(
            coords, energy, title=f"PILArNet-M-mini test event {event}: energy"
        ),
        "panda-event-labels": _comparison_figure(
            coords,
            motif,
            pid,
            left_title="Trajectory topology",
            right_title="Particle identity",
            names=MOTIF_NAMES,
            colors=MOTIF_COLORS,
            right_names=PID_NAMES,
            right_colors=PID_COLORS,
            title=f"PILArNet-M-mini test event {event}: point labels",
        ),
        "panda-event-particles": _instance_figure(
            coords,
            particle,
            title=f"PILArNet-M-mini test event {event}: particle instances",
        ),
        "panda-event-interactions": _instance_figure(
            coords,
            interaction,
            title=f"PILArNet-M-mini test event {event}: interaction instances",
        ),
    }
    for stem, figure in figures.items():
        _write_figure(figure, output_dir, stem)
        if show:
            figure.show()

    _write_static(
        coords,
        (("Trajectory topology", motif), ("Particle identity", pid)),
        output_dir / "panda-event-labels.png",
    )
    _write_static(
        coords,
        (("Particle instances", particle), ("Interaction instances", interaction)),
        output_dir / "panda-event-instances.png",
    )

    metadata = {
        "dataset": "DeepLearnPhysics/PILArNet-M-mini",
        "split": "test",
        "revision": "v2",
        "event": event,
        "num_points_after_transform": int(len(coords)),
        "motif_counts": {
            MOTIF_NAMES[i]: int(np.count_nonzero(motif == i)) for i in range(len(MOTIF_NAMES))
        },
        "pid_counts": {
            PID_NAMES[i]: int(np.count_nonzero(pid == i)) for i in range(len(PID_NAMES))
        },
        "particle_instances": int(np.count_nonzero(np.unique(particle) >= 0)),
        "interaction_instances": int(np.count_nonzero(np.unique(interaction) >= 0)),
    }
    (output_dir / "panda-event-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


# %% [markdown]
# ## Released model outputs
#
# These helpers run the four published Panda checkpoints.  Semantic and
# instance predictions stay aligned with the transformed event; base features
# are compressed to three display channels with PCA.

# %%
def _load_model(repo_id: str, *, device: str, disable_flash: bool):
    import pimm

    if not disable_flash:
        return pimm.from_pretrained(repo_id, device=device)

    # ``enable_flash`` appears at different nesting levels in the released
    # semantic/base and detector configs.  Override every occurrence before
    # construction, then load exactly the same weights.
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=("config.json", "model.safetensors", "model.bin"),
        )
    )
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    model_config = config.get("model", config)

    def recurse(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "enable_flash":
                    value[key] = False
                else:
                    recurse(child)
        elif isinstance(value, list):
            for child in value:
                recurse(child)

    recurse(model_config)
    return pimm.from_pretrained(snapshot, model_config=model_config, device=device)


def render_model_figures(
    sample: Mapping,
    output_dir: Path,
    *,
    models: Iterable[str],
    device: str,
    disable_flash: bool,
    show: bool = False,
):
    """Run requested Panda checkpoints and render their actual outputs."""
    import torch

    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "Released Panda inference currently needs a CUDA device because the "
            "PTv3 backbone uses spconv. Use --models dataset on CPU."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --models dataset on this machine.")

    coords = _as_numpy(sample["coord"])
    motif = _flatten(sample["segment_motif"]).astype(int)
    particle_truth = _flatten(sample["instance_particle"]).astype(int)
    interaction_truth = _flatten(sample["instance_interaction"]).astype(int)

    for name in models:
        if name == "dataset":
            continue
        model = _load_model(MODEL_REPOS[name], device=device, disable_flash=disable_flash)
        batch = _model_input(sample, device)
        with torch.inference_mode():
            output = model(copy.copy(batch))

        if name == "semantic":
            prediction = _as_numpy(output["seg_logits"].argmax(dim=1)).astype(int)
            figure = _comparison_figure(
                coords,
                motif,
                prediction,
                left_title="Truth",
                right_title="Panda prediction",
                names=MOTIF_NAMES,
                colors=MOTIF_COLORS,
                title="Panda semantic segmentation",
            )
            _write_figure(figure, output_dir, "panda-semantic")
            if show:
                figure.show()
            _write_static(
                coords,
                (("Truth", motif), ("Panda prediction", prediction)),
                output_dir / "panda-semantic.png",
            )
        elif name == "base":
            features = output.feat if hasattr(output, "feat") else output["point"].feat
            u, _, _ = torch.pca_lowrank(features.float(), q=3, niter=3, center=True)
            rgb = u - u.amin(dim=0)
            rgb = rgb / rgb.amax(dim=0).clamp_min(1e-12)
            rgb = _as_numpy(rgb)
            figure = _feature_figure(coords, rgb, title="Panda base features: PCA → RGB")
            _write_figure(figure, output_dir, "panda-base-features")
            if show:
                figure.show()
            _write_static(
                coords,
                (("Panda base features: PCA → RGB", rgb),),
                output_dir / "panda-base-features.png",
                categorical=False,
            )
        elif name in {"particle", "interaction"}:
            result = model.postprocess(output)
            prediction = _as_numpy(result["instance_labels"]).astype(int)
            truth = particle_truth if name == "particle" else interaction_truth
            figure = _comparison_figure(
                coords,
                truth,
                prediction,
                left_title="Truth",
                right_title="Panda prediction",
                names=tuple(f"Instance {i}" for i in range(128)),
                colors=INSTANCE_COLORS,
                title=f"Panda {name} clustering",
            )
            _write_figure(figure, output_dir, f"panda-{name}")
            if show:
                figure.show()
            _write_static(
                coords,
                (("Truth", truth), ("Panda prediction", prediction)),
                output_dir / f"panda-{name}.png",
            )


# %% [markdown]
# ## Load one mini event
#
# With no explicit data root, Hugging Face downloads only the mini test shard.
# Fixing NumPy's seed makes the representative chosen per occupied voxel stable.

# %%
def load_mini_event(*, event: int, data_root: Path | None, seed: int):
    """Download (if needed), transform, and return one deterministic mini event."""
    if data_root is None:
        from huggingface_hub import snapshot_download

        data_root = Path(
            snapshot_download(
                repo_id="DeepLearnPhysics/PILArNet-M-mini",
                repo_type="dataset",
                allow_patterns=("test/*.h5",),
            )
        )

    from pimm.datasets import build_dataset

    # GridSample(mode="train") chooses one representative in each occupied
    # voxel.  Fix NumPy's seed so the tutorial output is reproducible.
    np.random.seed(seed)
    dataset = build_dataset(dict(
        type="PILArNetH5Dataset",
        data_root=str(data_root),
        split="test",
        revision="v2",
        transform=TEST_TRANSFORM,
        energy_threshold=0.13,
        min_points=1024,
    ))
    if event < 0 or event >= len(dataset):
        raise IndexError(f"event must be in [0, {len(dataset) - 1}], got {event}")
    return dataset[event]


# %% [markdown]
# ## Command-line entry point
#
# This section is ignored when cells run inside Jupyter, but makes the same file
# usable as a normal script and keeps all figure-generation logic in one place.

# %%
def parse_args(argv: Sequence[str] | None = None):
    default_output = Path(__file__).resolve().parents[1] / "_static" / "tutorials"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=int, default=0, help="Mini test-event index")
    parser.add_argument("--seed", type=int, default=7, help="GridSample random seed")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Local PILArNet-M-mini root; downloaded from Hugging Face when omitted",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dataset"],
        choices=("dataset", "base", "semantic", "particle", "interaction", "all"),
        help="Figures to regenerate; anything except dataset currently needs CUDA",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-flash",
        action="store_true",
        help="Disable Flash Attention recursively in released model configs",
    )
    return parser.parse_args(argv)


def _find_repository_root(start: Path | None = None) -> Path:
    """Find the checkout root from either Jupyter's or a shell's working dir."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "docs" / "source"
        ).is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the pimm checkout above the current working directory. "
        "Start Jupyter inside the cloned repository or set the output path in "
        "the final cell."
    )


# %% [markdown]
# ## Run the selected cells
#
# When converted to ``.ipynb``, edit these three values and run all cells. The
# default is the CPU-only dataset exploration; add model names on a CUDA host.

# %%
NOTEBOOK_MODELS = {"dataset"}
NOTEBOOK_DEVICE = "cuda"
NOTEBOOK_DISABLE_FLASH = False


def _run(
    *,
    event: int,
    seed: int,
    data_root: Path | None,
    output_dir: Path,
    models: set[str],
    device: str,
    disable_flash: bool,
    show: bool,
):
    if "all" in models:
        models = {"dataset", "base", "semantic", "particle", "interaction"}

    sample = load_mini_event(event=event, data_root=data_root, seed=seed)
    metadata = render_dataset_figures(sample, output_dir, event=event, show=show)
    if models != {"dataset"}:
        render_model_figures(
            sample,
            output_dir,
            models=models,
            device=device,
            disable_flash=disable_flash,
            show=show,
        )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote figures to {output_dir.resolve()}")
    return sample, metadata


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    models = set(args.models)
    return _run(
        event=args.event,
        seed=args.seed,
        data_root=args.data_root,
        output_dir=args.output_dir,
        models=models,
        device=args.device,
        disable_flash=args.disable_flash,
        show=False,
    )


if "get_ipython" in globals():
    # Jupyter supplies its own command-line arguments, so do not call
    # ``parse_args`` here. Figures are also displayed inline.
    # A kernel may start either at the server root or beside the notebook, so
    # locate the checkout instead of assuming one particular working directory.
    _default_output = (
        _find_repository_root() / "docs" / "source" / "_static" / "tutorials"
    )
    sample, metadata = _run(
        event=0,
        seed=7,
        data_root=None,
        output_dir=_default_output,
        models=set(NOTEBOOK_MODELS),
        device=NOTEBOOK_DEVICE,
        disable_flash=NOTEBOOK_DISABLE_FLASH,
        show=True,
    )
elif __name__ == "__main__":
    main()
