"""Regenerate the figures used by the Explore PoLAr-MAE documentation.

This file is a Jupytext-compatible ``percent`` notebook.  It can be executed as
ordinary Python, opened cell-by-cell in VS Code, or converted to ``.ipynb``.
The dataset view runs on CPU; released-model inference may be run on CPU or
CUDA, although CUDA is substantially faster.
"""

# %% [markdown]
# # Explore PoLAr-MAE, reproducibly
#
# Download one event from PILArNet-M-mini, apply each released checkpoint's
# preprocessing contract, and write responsive Plotly figures.  The final cell
# selects which model outputs to regenerate.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CLASS_NAMES = ("Shower", "Track", "Michel", "Delta")
CLASS_COLORS = ("#79b5a4", "#f5d69b", "#185890", "#ba4a09")

SEMANTIC_TRANSFORM = [
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

PRETRAIN_TRANSFORM = [
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "energy"),
        feat_keys=("coord", "energy"),
    ),
]


# %% [markdown]
# ## Figure helpers

# %%
def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _scene_style(fig, *, title: str):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
    )
    fig.update_scenes(
        aspectmode="data",
        xaxis=dict(title="x", showbackground=False),
        yaxis=dict(title="y", showbackground=False),
        zaxis=dict(title="z", showbackground=False),
        camera=dict(eye=dict(x=1.45, y=1.45, z=1.1)),
    )
    return fig


def _class_traces(coords, labels, *, scene: str, legendgroup: str):
    import plotly.graph_objects as go

    traces = []
    for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        selected = labels == class_id
        if not np.any(selected):
            continue
        traces.append(
            go.Scatter3d(
                x=coords[selected, 0],
                y=coords[selected, 1],
                z=coords[selected, 2],
                mode="markers",
                marker=dict(size=2.7, color=color, opacity=0.9),
                name=f"{name} ({int(selected.sum()):,})",
                legendgroup=f"{legendgroup}-{class_id}",
                scene=scene,
                hovertemplate=(
                    f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}"
                    "<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )
    return traces


def semantic_figure(coords, truth, prediction=None):
    from plotly.subplots import make_subplots

    if prediction is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
        for trace in _class_traces(coords, truth, scene="scene", legendgroup="truth"):
            fig.add_trace(trace, row=1, col=1)
        return _scene_style(fig, title="PILArNet-M-mini: semantic truth")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("PILArNet-M truth", "PoLAr-MAE prediction"),
        horizontal_spacing=0.02,
    )
    for trace in _class_traces(coords, truth, scene="scene", legendgroup="truth"):
        fig.add_trace(trace, row=1, col=1)
    for trace in _class_traces(
        coords, prediction, scene="scene2", legendgroup="prediction"
    ):
        fig.add_trace(trace, row=1, col=2)
    return _scene_style(fig, title="PoLAr-MAE semantic segmentation")


def reconstruction_figure(visible, predicted, target):
    import plotly.graph_objects as go

    fig = go.Figure()
    for points, name, color, opacity, visible_by_default in (
        (visible, "Visible groups", "#334155", 0.5, True),
        (predicted, "Reconstructed masked groups", "#e8684a", 0.85, True),
        (target, "Masked targets", "#1f6feb", 0.45, "legendonly"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2.7, color=color, opacity=opacity),
                name=name,
                visible=visible_by_default,
                hovertemplate=(
                    f"{name}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}"
                    "<br>z=%{z:.4f}<extra></extra>"
                ),
            )
        )
    return _scene_style(fig, title="PoLAr-MAE masked-group reconstruction")


def _rgb(values):
    values = values - values.min(axis=0, keepdims=True)
    values = values / np.maximum(values.max(axis=0, keepdims=True), 1e-8)
    values = np.clip(np.rint(values * 255), 0, 255).astype(np.uint8)
    return [f"rgb({red},{green},{blue})" for red, green, blue in values]


def representation_figure(centers, encoded_rgb, residual_rgb):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Token PCA", "PCA after linear position regression"),
        horizontal_spacing=0.02,
    )
    for column, colors, name in (
        (1, encoded_rgb, "Token PCA"),
        (2, residual_rgb, "Position-residual PCA"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker=dict(size=5.5, color=_rgb(colors), opacity=0.9),
                name=name,
                showlegend=False,
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    return _scene_style(fig, title="PoLAr-MAE token representations")


def _write_html(fig, output_dir: Path, stem: str, *, show: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_dir / f"{stem}.html",
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
    if show:
        fig.show()


def _write_semantic_png(coords, truth, prediction, output: Path):
    """Write a no-JavaScript fallback without requiring Kaleido."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5.4))
    for column, (labels, title) in enumerate(
        ((truth, "PILArNet-M truth"), (prediction, "PoLAr-MAE prediction")), 1
    ):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        for class_id, color in enumerate(CLASS_COLORS):
            selected = labels == class_id
            ax.scatter(
                coords[selected, 0], coords[selected, 1], coords[selected, 2],
                c=color, s=1.2,
            )
        ax.set(title=title, xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect(np.ptp(coords, axis=0))
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# %% [markdown]
# ## Load one event in both preprocessing contracts

# %%
def load_views(*, event: int, data_root: Path | None):
    if data_root is None:
        from huggingface_hub import snapshot_download

        data_root = Path(
            snapshot_download(
                repo_id="DeepLearnPhysics/PILArNet-M-mini",
                repo_type="dataset",
                allow_patterns=("test/*.h5",),
            )
        )

    from pimm.datasets import build_dataset, collate_fn

    common = dict(
        type="PILArNetH5Dataset",
        data_root=str(data_root),
        revision="v2",
        split="test",
        energy_threshold=0.13,
        remove_low_energy_scatters=True,
        min_points=0,
    )
    semantic_dataset = build_dataset(dict(**common, transform=SEMANTIC_TRANSFORM))
    pretrain_dataset = build_dataset(dict(**common, transform=PRETRAIN_TRANSFORM))
    if event < 0 or event >= len(semantic_dataset):
        raise IndexError(f"event must be in [0, {len(semantic_dataset) - 1}], got {event}")
    return (
        collate_fn([semantic_dataset[event]]),
        collate_fn([pretrain_dataset[event]]),
    )


def _to_device(batch: Mapping, device: str):
    import torch

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# %% [markdown]
# ## Semantic inference

# %%
def render_semantic(
    batch, output_dir: Path, *, device: str, seed: int, show: bool
):
    import torch
    import pimm

    batch = _to_device(batch, device)
    model_input = {key: batch[key] for key in ("coord", "feat", "offset")}
    model = pimm.from_pretrained(
        "DeepLearnPhysics/PoLAr-MAE-Semantic", device=device
    )
    # Token-center FPS uses a random starting point even in evaluation mode.
    # Fix it so checked-in figures and counts can be regenerated exactly.
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model(model_input)

    logits = output["seg_logits"]
    prediction = _numpy(logits.argmax(dim=-1)).astype(int)
    truth = _numpy(batch["segment"]).reshape(-1).astype(int)
    coords = _numpy(batch["coord"])
    figure = semantic_figure(coords, truth, prediction)
    _write_html(figure, output_dir, "polarmae-semantic", show=show)
    _write_semantic_png(
        coords, truth, prediction, output_dir / "polarmae-semantic.png"
    )
    return {
        "logits_shape": list(logits.shape),
        "prediction_counts": np.bincount(prediction, minlength=4).tolist(),
        "truth_counts": np.bincount(truth, minlength=5).tolist(),
    }


# %% [markdown]
# ## Masked reconstruction and encoder representations

# %%
def _gather(x, indices):
    import torch

    return torch.gather(
        x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[2])
    )


def _gather_groups(x, indices):
    import torch

    return torch.gather(
        x,
        1,
        indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, x.shape[2], x.shape[3]
        ),
    )


def trace_pretrain_model(model, batch):
    """Run the public submodules once and retain arrays needed for the figures."""
    import torch
    import torch.nn.functional as F

    from pimm.models.losses.chamfer import chamfer_distance
    from pimm.models.polarmae.data import packed_to_batched

    feat, offset = batch["feat"], batch["offset"]
    points, lengths = packed_to_batched(feat, offset)
    grouped = model.grouping(points, lengths)
    groups = grouped["groups"]
    centers = grouped["centers"]
    embedding_mask = grouped["embedding_mask"]
    point_mask = grouped["point_mask"]

    masked_idx, masked_mask, visible_idx, visible_mask = model.masking(
        embedding_mask.sum(-1)
    )
    visible_groups = _gather_groups(groups, visible_idx)
    visible_point_mask = _gather(point_mask, visible_idx)
    visible_centers = _gather(centers, visible_idx)
    masked_groups = _gather_groups(groups, masked_idx)
    masked_point_mask = _gather(point_mask, masked_idx) * masked_mask.unsqueeze(-1)
    masked_centers = _gather(centers, masked_idx)

    valid_tokens = model.embedding(
        visible_groups[visible_mask], visible_point_mask[visible_mask].unsqueeze(1)
    )
    tokens = visible_groups.new_zeros(
        visible_groups.shape[0], visible_groups.shape[1], valid_tokens.shape[-1]
    )
    tokens[visible_mask] = valid_tokens
    visible_position = model.pos_embed(visible_centers)
    encoded = model.encoder(tokens, visible_position, visible_mask).last_hidden_state

    masked_position = model.pos_embed(masked_centers)
    mask_tokens = model.mask_token.expand(
        masked_mask.shape[0], masked_mask.shape[1], -1
    )
    # Released checkpoints exist in both decoder forms.  Cross-attention uses
    # visible tokens as ``kv``; the original self-attention decoder concatenates
    # visible and masked tokens before decoding.
    decoder_uses_kv = model.decoder.blocks[0].norm1_kv is not None
    if decoder_uses_kv:
        decoded_masked = model.decoder(
            mask_tokens,
            masked_position,
            masked_mask,
            kv=encoded,
            pos_kv=visible_position,
            kv_mask=visible_mask,
        ).last_hidden_state
    else:
        visible_width = encoded.shape[1]
        decoder_tokens = torch.cat((encoded, mask_tokens), dim=1)
        decoder_position = torch.cat((visible_position, masked_position), dim=1)
        decoder_mask = torch.cat((visible_mask, masked_mask), dim=1)
        decoded = model.decoder(
            decoder_tokens,
            decoder_position,
            decoder_mask,
        ).last_hidden_state
        decoded_masked = decoded[:, visible_width:]

    masked_output = decoded_masked[masked_mask]
    masked_groups_flat = masked_groups[masked_mask]
    masked_point_mask_flat = masked_point_mask[masked_mask]
    point_lengths = masked_point_mask_flat.sum(-1)
    predicted = model.increase_dim(masked_output.unsqueeze(-1)).squeeze(-1)
    predicted = predicted.reshape(predicted.shape[0], -1, model.mae_channels)
    chamfer_loss, _, _ = chamfer_distance(
        predicted.float(),
        masked_groups_flat[..., : model.mae_channels].float(),
        x_lengths=point_lengths,
        y_lengths=point_lengths,
    )

    energy_features = model.equivariant_patch_encoder(
        masked_groups_flat[..., :3], masked_point_mask_flat.unsqueeze(1)
    )
    energy_input = torch.cat((energy_features, masked_output), dim=1)
    predicted_energy = model.energy_decoder(
        energy_input.unsqueeze(-1)
    ).squeeze(-1)
    energy_loss = F.mse_loss(
        predicted_energy[masked_point_mask_flat].float(),
        masked_groups_flat[masked_point_mask_flat][..., -1].float(),
    )
    total = sum(
        model.loss_weights.get(name, 1.0) * value
        for name, value in (("chamfer", chamfer_loss), ("energy", energy_loss))
    )

    radius = float(model.grouping.rescale_by_group_radius or 1.0)
    visible_world = (
        visible_groups[..., :3] * radius + visible_centers.unsqueeze(2)
    )[visible_point_mask & visible_mask.unsqueeze(-1)]
    masked_centers_flat = masked_centers[masked_mask]
    predicted_world = predicted[..., :3] * radius + masked_centers_flat.unsqueeze(1)
    target_world = (
        masked_groups_flat[..., :3] * radius + masked_centers_flat.unsqueeze(1)
    )

    return {
        "loss": total,
        "chamfer_loss": chamfer_loss,
        "energy_loss": energy_loss,
        "visible": visible_world,
        "predicted": predicted_world[masked_point_mask_flat],
        "target": target_world[masked_point_mask_flat],
    }


def encode_all_tokens(model, batch):
    import torch
    from pimm.models.polarmae.data import packed_to_batched

    points, lengths = packed_to_batched(batch["feat"], batch["offset"])
    grouped = model.grouping(points, lengths)
    groups = grouped["groups"]
    centers = grouped["centers"]
    token_mask = grouped["embedding_mask"]
    point_mask = grouped["point_mask"]

    valid_tokens = model.embedding(
        groups[token_mask], point_mask[token_mask].unsqueeze(1)
    )
    tokens = groups.new_zeros(
        groups.shape[0], groups.shape[1], valid_tokens.shape[-1]
    )
    tokens[token_mask] = valid_tokens
    encoded = model.encoder(
        tokens, model.pos_embed(centers), token_mask
    ).last_hidden_state

    event_tokens = encoded[0, token_mask[0]].float()
    event_centers = centers[0, token_mask[0], :3].float()
    _, _, axes = torch.pca_lowrank(event_tokens, q=3, center=True)
    encoded_rgb = event_tokens @ axes[:, :3]

    design = torch.cat(
        (torch.ones(len(event_centers), 1, device=event_centers.device), event_centers),
        dim=1,
    )
    coefficients = torch.linalg.lstsq(design, event_tokens).solution
    residual = event_tokens - design @ coefficients
    _, _, residual_axes = torch.pca_lowrank(residual, q=3, center=True)
    residual_rgb = residual @ residual_axes[:, :3]
    return event_centers, encoded_rgb, residual_rgb


def render_pretrain(batch, output_dir: Path, *, device: str, seed: int, show: bool):
    import torch
    import pimm

    batch = _to_device(batch, device)
    model = pimm.from_pretrained(
        "DeepLearnPhysics/PoLAr-MAE-Pretrain", device=device
    )
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        traced = trace_pretrain_model(model, batch)
        centers, encoded_rgb, residual_rgb = encode_all_tokens(model, batch)

    reconstruction = reconstruction_figure(
        _numpy(traced["visible"]),
        _numpy(traced["predicted"]),
        _numpy(traced["target"]),
    )
    _write_html(
        reconstruction, output_dir, "polarmae-reconstruction", show=show
    )
    representations = representation_figure(
        _numpy(centers), _numpy(encoded_rgb), _numpy(residual_rgb)
    )
    _write_html(
        representations, output_dir, "polarmae-representations", show=show
    )
    return {
        name: float(traced[name].detach().cpu())
        for name in ("loss", "chamfer_loss", "energy_loss")
    }


# %% [markdown]
# ## Run selected outputs

# %%
def _find_repository_root(start: Path | None = None):
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "docs" / "source"
        ).is_dir():
            return candidate
    raise RuntimeError("Start Jupyter inside the pimm checkout.")


def _run(
    *,
    event: int,
    data_root: Path | None,
    output_dir: Path,
    models: set[str],
    device: str,
    seed: int,
    show: bool,
):
    if "all" in models:
        models = {"dataset", "semantic", "pretrain"}
    semantic_batch, pretrain_batch = load_views(event=event, data_root=data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = _numpy(semantic_batch["coord"])
    truth = _numpy(semantic_batch["segment"]).reshape(-1).astype(int)
    if "dataset" in models:
        _write_html(
            semantic_figure(coords, truth),
            output_dir,
            "polarmae-event-truth",
            show=show,
        )

    metadata = {
        "dataset": "DeepLearnPhysics/PILArNet-M-mini",
        "split": "test",
        "revision": "v2",
        "event": event,
        "num_points": int(len(coords)),
    }
    if "semantic" in models:
        metadata["semantic"] = render_semantic(
            semantic_batch, output_dir, device=device, seed=seed, show=show
        )
    if "pretrain" in models:
        metadata["pretrain"] = render_pretrain(
            pretrain_batch,
            output_dir,
            device=device,
            seed=seed,
            show=show,
        )
    (output_dir / "polarmae-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote figures to {output_dir.resolve()}")
    return semantic_batch, pretrain_batch, metadata


def parse_args(argv: Sequence[str] | None = None):
    default_output = Path(__file__).resolve().parents[1] / "_static" / "tutorials"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dataset"],
        choices=("dataset", "semantic", "pretrain", "all"),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Model device; use cpu for a slower GPU-free run",
    )
    return parser.parse_args(argv)


NOTEBOOK_MODELS = {"dataset"}
NOTEBOOK_DEVICE = "cuda"


if "get_ipython" in globals():
    _output = (
        _find_repository_root() / "docs" / "source" / "_static" / "tutorials"
    )
    semantic_batch, pretrain_batch, metadata = _run(
        event=0,
        data_root=None,
        output_dir=_output,
        models=set(NOTEBOOK_MODELS),
        device=NOTEBOOK_DEVICE,
        seed=7,
        show=True,
    )
elif __name__ == "__main__":
    args = parse_args()
    _run(
        event=args.event,
        data_root=args.data_root,
        output_dir=args.output_dir,
        models=set(args.models),
        device=args.device,
        seed=args.seed,
        show=False,
    )
