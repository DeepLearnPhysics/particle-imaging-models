---
sd_hide_title: true
---

# pimm documentation

:::{div} pimm-hero

```{image} _static/logo.svg
:alt: pimm logo
:class: pimm-hero-logo
:width: 96px
```

# Particle Imaging Models

[Foundation-model research for particle-imaging detectors.]{.pimm-tagline}

```{button-ref} getting_started/installation
:ref-type: myst
:color: primary
:class: sd-px-4
Install pimm
```
```{button-ref} getting_started/quickstart
:ref-type: myst
:color: secondary
:outline:
:class: sd-px-4
Run the quickstart
```

:::

pimm is research infrastructure for variable-length, three-dimensional point
clouds from particle-imaging detectors. It provides model families, datasets,
training and evaluation loops, distributed launchers, and portable exports.

## Install

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

The installer creates the locked environment and checks the native operators.
See {doc}`Installation <getting_started/installation>` for requirements,
containers, manual setup, and hardware-specific settings.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pimm-card-grid

:::{grid-item-card} Use a trained model
:link: models/pretrained
:link-type: doc
Load a local export or a published checkpoint, reproduce its preprocessing,
and run a forward pass.
:::

:::{grid-item-card} Fine-tune a backbone
:link: workflows/fine_tune
:link-type: doc
Select a checkpoint, map its weights into a task model, freeze or unfreeze the
right parameters, and evaluate the result.
:::

:::{grid-item-card} Train or pretrain
:link: workflows/train
:link-type: doc
Choose a real recipe, set the data root, make a small smoke run, then scale it
without changing the training config.
:::

:::{grid-item-card} Bring your own data
:link: data/custom
:link-type: doc
Implement a dataset that emits pimm's packed sample contract, validate one
batch, and register it for use in a config.
:::

:::{grid-item-card} Run on a cluster
:link: workflows/slurm
:link-type: doc
Describe your Slurm site once, dry-run the generated job, submit it, monitor it,
and resume safely.
:::

:::{grid-item-card} Extend pimm
:link: extend/architecture
:link-type: doc
Trace one batch through registries, transforms, model, trainer, hooks, evaluator,
and checkpoint code before adding a component.
:::

::::

## Verify an installation

This command resolves the launcher and the bundled tiny configuration, then
prints the command it would run. It does not read data or start training.

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  --dry-run
```

For an actual training run, follow the {doc}`ten-minute quickstart
<getting_started/quickstart>`. Training requires Linux x86-64 and an NVIDIA
driver compatible with the locked CUDA 12.6 environment.

## The experiment lifecycle

<div class="pimm-pipeline" role="list" aria-label="pimm experiment lifecycle">
  <span role="listitem">Data + transforms</span>
  <span role="listitem">Packed batch</span>
  <span role="listitem">Model + loss</span>
  <span role="listitem">Trainer + hooks</span>
  <span role="listitem">Checkpoint + metrics</span>
  <span role="listitem">Export + inference</span>
</div>

The Python config defines *what* is trained. Launch YAML and launcher flags
define *where* it runs. Every run records the resolved config, source snapshot,
command metadata, logs, and checkpoint state together. The {doc}`experiment
anatomy <getting_started/concepts>` page connects those objects to their exact
implementation.

## Supported scope

| Need | Included |
|---|---|
| Sparse backbones | Point Transformer v1/v2/v3, SparseUNet, LitePT, Volt |
| Representation learning | Panda/Sonata, PoLAr-MAE |
| Downstream tasks | semantic segmentation, PointGroup, Panda Detector |
| Scale | local `torchrun`, DDP, experimental FSDP2 paths, Slurm via Submitit |
| Portability | structured checkpoints, plain model weights, Hugging Face exports |

:::{important}
pimm currently supports 3D sparse point clouds. A model is only scientifically
reusable when its coordinate convention, units, feature order, transforms,
dataset revision, and checkpoint revision are known. Start with {doc}`Data
conventions <data/conventions>` and the {doc}`model chooser <models/index>`;
do not infer preprocessing from an architecture name.
:::

## Need an exact answer?

- {doc}`CLI reference <reference/cli>` — launcher, submitter, and export syntax.
- {doc}`Configuration reference <reference/configuration>` — precedence,
  inheritance, and common keys.
- {doc}`Checkpoint semantics <operations/checkpoints>` — what is saved and
  when a resume is exact.
- {doc}`Troubleshooting <operations/troubleshooting>` — symptoms, diagnostics,
  and fixes.
- {doc}`Python API <api/index>` — generated classes, functions, and registries.

```{toctree}
:hidden:
:maxdepth: 2

Start <getting_started/index>
Workflows <workflows/index>
Models <models/index>
Data <data/index>
Run & reproduce <operations/index>
Tutorials <tutorials/index>
Extend <extend/index>
Reference <reference/index>
Project <project/index>
```
