#!/usr/bin/env python
"""Generate per-registry API reference pages for the pimm docs.

For every pimm registry (MODELS, DATASETS, TRANSFORMS, HOOKS, LOSSES, TRAINERS)
this writes a reStructuredText page under ``source/api/registry/``. Each page is
split into curated **sections** (e.g. Backbones / Pretraining / ... for models;
Checkpointing / Logging / Evaluation / ... for hooks) rather than one flat list,
and each section contains:

* a table mapping each config ``type=`` string to its class, and
* an ``autosummary`` block (with ``:toctree:``) so Sphinx autodoc generates a
  full page per class straight from the source docstrings.

Sections are defined by ``CATEGORIES`` below; anything that matches no section
falls into a trailing "Other" section (and prints a warning) so new classes are
never silently dropped.

Run automatically by ``make html`` (the ``gen`` target). It imports ``pimm``,
so run it from the project environment.
"""
from __future__ import annotations

import inspect
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "source" / "api" / "registry"


# --- section matchers -------------------------------------------------------

def _comps(module: str) -> set:
    """Dotted components of a module path, as a set (collision-free matching)."""
    return set((module or "").split("."))


def mod_in(*families):
    """Match when any ``family`` is a dotted component of the class's module."""
    fams = set(families)

    def pred(names, module):
        return bool(fams & _comps(module))

    return pred


def name_in(*registered):
    """Match when any of the class's registered ``type`` names is in the set."""
    want = set(registered)

    def pred(names, module):
        return bool(want & set(names))

    return pred


def name_prefix(prefix):
    """Match when any registered name starts with ``prefix``."""

    def pred(names, module):
        return any(n.startswith(prefix) for n in names)

    return pred


# slug -> ordered list of (section title, one-line description, predicate)
CATEGORIES = {
    "models": [
        ("Backbones",
         "Encoder/decoder networks used as a ``backbone=`` inside task models.",
         mod_in("point_transformer_v3", "point_transformer_v2", "sparse_unet", "litept", "volt")),
        ("Self-supervised pretraining",
         "Discriminative and masked-autoencoder pretraining systems.",
         mod_in("sonata", "polarmae", "distill")),
        ("Segmentation & classification",
         "Per-point semantic segmentation and event/point classification heads.",
         mod_in("default", "classifier", "point_transformer")),
        ("Object detection",
         "Instance / panoptic detectors: Mask2Former-style Panda, autoregressive Panda, and PointGroup.",
         mod_in("panda_detector", "panda_ar", "point_group")),
        ("Adapters",
         "Parameter-efficient adapters that wrap an existing model.",
         mod_in("lora")),
    ],
    "hooks": [
        ("Checkpointing",
         "Save and load checkpoints / resume state (delegate to ``CheckpointManager``).",
         mod_in("checkpoint")),
        ("Logging",
         "Per-step / per-epoch scalar logging and run naming.",
         mod_in("logging")),
        ("Evaluation",
         "In-loop validators, final testers, and SSL probes; they set the checkpoint-selection metric.",
         mod_in("eval")),
        ("Diagnostics",
         "Gradient/feature/prototype/parameter monitors and dtype/anneal controls.",
         mod_in("diagnostics")),
        ("Optimizer schedules",
         "Hooks that mutate optimizer parameter groups (weight-decay exclusion / scheduling).",
         mod_in("optimizer")),
        ("Hugging Face export",
         "Push checkpoints to the Hub during or after training.",
         mod_in("export")),
        ("Resources & profiling",
         "Memory/GC management, resource-utilization logging, and the runtime profiler.",
         mod_in("resources", "profiling")),
        ("Core",
         "Lifecycle plumbing shared by other hooks.",
         mod_in("default")),
    ],
    "transforms": [
        ("Input & collection",
         "Tensor conversion and the final ``Collect`` projection into the model batch.",
         mod_in("base")),
        ("Spatial & geometric",
         "Coordinate normalization, grid sampling, crops, rotations, jitter, and warps.",
         mod_in("spatial")),
        ("Feature, energy & color",
         "Energy/charge log-transforms and feature/color augmentations.",
         mod_in("color")),
        ("Detector-specific",
         "LArTPC label/motif derivations (PDG→semantic, stuff/noise, multi-scale time).",
         mod_in("detector")),
        ("Instance",
         "Instance parsing, anchors, and local-covariance features for detection.",
         mod_in("instance")),
        ("Multi-view (SSL)",
         "Contrastive / multi-scale view generators for self-supervised pretraining.",
         mod_in("multiview")),
        ("Rasterization (ring images)",
         "Cylinder-unrolling and ring rasterization for Water-Cherenkov images.",
         mod_in("rasterize")),
    ],
    "losses": [
        ("Classification",
         "Cross-entropy and focal variants for class logits.",
         name_in("CrossEntropyLoss", "SmoothCELoss", "FocalLoss", "BinaryFocalLoss",
                 "CrossEntropyHeadLoss")),
        ("Segmentation",
         "Region-overlap losses for semantic segmentation.",
         name_in("DiceLoss", "LovaszLoss")),
        ("Regression",
         "Continuous-target losses (momentum, vertex, ...).",
         name_in("L1RegressionLoss", "MSERegressionLoss", "SmoothL1RegressionLoss")),
        ("Instance segmentation",
         "Set-based mask + class (+ regression) losses with Hungarian matching.",
         mod_in("instance_fast", "instance_regression_fast", "instance_unified_fast")),
    ],
    "datasets": [
        ("Generic",
         "Directory / JSON-split datasets of pre-arranged numpy assets.",
         mod_in("defaults")),
        ("LArTPC point clouds",
         "Liquid-argon TPC detector datasets (PILArNet-M, JAXTPC, MicroBooNE).",
         name_in("PILArNetH5Dataset", "JAXTPCDataset", "UBooNEH5Dataset")),
        ("Water Cherenkov (LUCiD)",
         "Water-Cherenkov sensor/segment datasets and SSL/ring-panoptic variants.",
         name_prefix("LUCiD")),
    ],
    "trainers": [
        ("General",
         "The default training loop and multi-dataset variant.",
         name_in("DefaultTrainer", "MultiDatasetTrainer")),
        ("Task-specific",
         "Trainers with task-specific collation / loaders.",
         name_in("InsegTrainer", "ImageClassTrainer")),
        ("Reinforcement learning",
         "Rollout-based policy optimization (GRPO).",
         name_in("GRPOTrainer")),
    ],
}


def _load_registries():
    import pimm  # noqa: F401  (side-effect registrations)
    import pimm.models  # noqa: F401
    import pimm.datasets  # noqa: F401
    import pimm.engines.train  # noqa: F401
    import pimm.engines.hooks  # noqa: F401

    from pimm.datasets.builder import DATASETS
    from pimm.datasets.transform.common import TRANSFORMS
    from pimm.engines.hooks.builder import HOOKS
    from pimm.engines.train import TRAINERS
    from pimm.models.builder import MODELS
    from pimm.models.losses.builder import LOSSES

    # slug, title, registry, build-key, intro
    return [
        ("models", "Models", MODELS, "model = dict(type=...)",
         "Every model and backbone buildable from a ``model = dict(type=...)`` "
         "config block."),
        ("datasets", "Datasets", DATASETS, "data.train = dict(type=...)",
         "Dataset classes buildable under ``data.train`` / ``data.val`` / "
         "``data.test``. See :doc:`../../data/index`."),
        ("transforms", "Transforms", TRANSFORMS, "transform=[dict(type=...)]",
         "Transform pipeline steps. See :doc:`../../data/transforms`."),
        ("hooks", "Hooks", HOOKS, "hooks=[dict(type=...)]",
         "Training lifecycle hooks. See :doc:`../../extend/add_hook`."),
        ("losses", "Losses", LOSSES, "criteria=[dict(type=...)]",
         "Loss functions assembled by "
         ":func:`~pimm.models.losses.builder.build_criteria`. See "
         ":doc:`../../extend/add_model`."),
        ("trainers", "Trainers", TRAINERS, "train = dict(type=...)",
         "Trainer classes selected by ``train.type``. See "
         ":doc:`../../extend/architecture`."),
    ]


def _module_dict(reg):
    return getattr(reg, "_module_dict", None) or getattr(reg, "module_dict", {})


def _summary(obj) -> str:
    doc = obj.__doc__ or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return " - "


def _qualpath(obj) -> str:
    mod = getattr(obj, "__module__", None)
    qn = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
    return f"{mod}.{qn}" if mod and qn else None


def _rst_escape(text: str) -> str:
    return text.replace("*", r"\*").replace("|", r"\|")


def _unique_rows(reg):
    """Return [(sorted names, qualpath, obj)] for unique registered classes."""
    entries = _module_dict(reg)
    by_class: dict[object, list[str]] = {}
    for name, obj in entries.items():
        by_class.setdefault(obj, []).append(name)
    rows = []
    for obj, names in by_class.items():
        path = _qualpath(obj)
        if path is None:
            continue
        rows.append((sorted(names), path, obj))
    rows.sort(key=lambda r: r[0][0].lower())
    return rows, len(entries)


def _section_for(names, module, categories):
    for title, _desc, pred in categories:
        if pred(names, module):
            return title
    return None


def _emit_section(lines, title, desc, rows):
    lines += [title, "-" * len(title), ""]
    if desc:
        lines += [desc, ""]
    # Visible table: the config ``type`` Name (linked to the class's API page)
    # and a one-line summary. The link text is the registry name you put in a
    # config; it points at the autodoc page generated below.
    lines += [".. list-table::", "   :header-rows: 1", "   :widths: 38 62", ""]
    lines += ["   * - Name", "     - Summary"]
    for names, path, obj in rows:
        name_links = " / ".join(f":class:`{n} <{path}>`" for n in names)
        lines += [
            f"   * - {name_links}",
            f"     - {_rst_escape(_summary(obj))}",
        ]
    lines += [""]
    # Hidden autosummary: generates one autodoc page per class (from the source
    # docstrings) for the links above to point at, without rendering its own
    # redundant table of full dotted paths.
    lines += [
        ".. container:: hidden-autosummary",
        "",
        "   .. autosummary::",
        "      :toctree: generated",
        "      :template: pimm_class.rst",
        "      :nosignatures:",
        "",
    ]
    for names, path, obj in rows:
        lines += [f"      {path}"]
    lines += [""]


def write_registry_page(slug, title, reg, build_key, intro):
    rows, n_names = _unique_rows(reg)
    categories = CATEGORIES.get(slug, [])

    # Bucket rows into sections (first match wins); collect the leftovers.
    buckets = {t: [] for (t, _d, _p) in categories}
    other = []
    for names, path, obj in rows:
        sect = _section_for(names, getattr(obj, "__module__", ""), categories)
        (buckets[sect] if sect else other).append((names, path, obj))

    lines = []
    head = f"{title} registry"
    lines += [head, "=" * len(head), ""]
    lines += [intro, ""]
    lines += [
        f"Build any of these from a config with ``{build_key}``, using the "
        "``type`` string in the first column. **Generated from the live "
        f"registry** - {len(rows)} classes ({n_names} names including aliases), "
        "grouped by role.",
        "",
    ]

    for title_s, desc_s, _pred in categories:
        sect_rows = buckets[title_s]
        if sect_rows:
            _emit_section(lines, title_s, desc_s, sect_rows)

    if other:
        names_list = ", ".join(sorted(n for r in other for n in r[0]))
        print(f"  [warn] {slug}: {len(other)} uncategorized -> 'Other': {names_list}")
        _emit_section(
            lines, "Other",
            "Registered classes not yet assigned to a section above "
            "(update ``CATEGORIES`` in ``docs/gen_api.py``).",
            other,
        )

    (OUT / f"{slug}.rst").write_text("\n".join(lines))
    return slug, title, len(rows)


def write_index(items):
    lines = [
        "Registries",
        "==========",
        "",
        "pimm assembles models, datasets, transforms, hooks, losses, and trainers",
        "from config dictionaries with a ``type`` key, resolved through small",
        "registries (see :doc:`../../getting_started/concepts`). These pages are",
        "**generated from the live registries and the source docstrings** - each",
        "lists every registered ``type``, grouped by role, and links to its autodoc",
        "page.",
        "",
    ]
    for slug, title, n in items:
        lines += [f"- :doc:`{title} <{slug}>` - {n} registered entries."]
    lines += ["", ".. toctree::", "   :hidden:", ""]
    for slug, _title, _n in items:
        lines += [f"   {slug}"]
    lines += [""]
    (OUT / "index.rst").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for slug, title, reg, build_key, intro in _load_registries():
        items.append(write_registry_page(slug, title, reg, build_key, intro))
        print(f"  generated api/registry/{slug}.rst  ({items[-1][2]} classes)")
    # Note: api/index.rst toctrees the per-registry pages directly (no
    # "Registries" wrapper page), so registry/index.rst is intentionally not
    # generated.


if __name__ == "__main__":
    main()
