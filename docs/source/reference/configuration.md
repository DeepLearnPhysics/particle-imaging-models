# Configuration reference

pimm has two configuration systems with different responsibilities:

| System | Location | Controls |
|---|---|---|
| experiment config | Python under `configs/` | data, transforms, model, loss, optimization, hooks, evaluation |
| launch config | YAML under `launch/` plus typed flags | site, resources, paths, container, Slurm, naming, resume handoff |

The practical guide to inheritance, list replacement, common fields, and
overrides is {doc}`Configuration <../operations/configuration>`. This page is
the catalog.

## Resolution order

```text
experiment: base Python config(s) → child Python config → post-`--` overrides
launch:     launch/defaults.yaml → site YAML → recipe YAML → CLI flags
```

Python dictionaries merge recursively; lists and scalars replace inherited
values. Launch YAML mappings merge recursively. A launch recipe path is passed
as `--recipe launch/runs/<name>.yaml`.

## Find and inspect a config

Config references omit the `configs/` prefix and `.py` suffix:

```bash
find configs -name '*.py' -not -path '*/_base_/*' | sort

uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --resources.nproc-per-node 1 \
  --dry-run
```

A dry run checks that inheritance resolves and the path exists, but it does not
build the model or open the dataset.

## Filename vocabulary

| Token | Meaning in the committed Panda task configs |
|---|---|
| `pretrain` | self-supervised or reconstruction training |
| `semseg` | point-wise semantic segmentation |
| `panseg` | query-based particle/interaction instance task |
| `pid` | particle-instance and particle-class targets |
| `vtx` | interaction-instance/vertex-oriented targets |
| `fft` | full fine-tuning of the pretrained backbone and task head |
| `dec` | decoder/head adaptation with the encoder frozen |
| `lin` | task head with the backbone frozen |
| `scratch` | the corresponding full-training config intended to run without a supplied warm-start weight |
| `reproduce` | evaluation-oriented config intended for a named released checkpoint |

These tokens describe config intent, not a verified result. Read the resolved
model, loader, weight, and hooks before using a filename in a paper.

The current `scratch` child files change only the W&B naming override; they do
not reject `--train.weight`. Confirm that the resolved `weight` is empty when a
from-scratch comparison matters.

## Committed experiment catalog

The paths below were checked by loading them through
{py:meth}`~pimm.utils.config.Config.fromfile` in this checkout. “Research
recipe” means committed and loadable, not benchmarked or API-stable.

### First-run and recovery fixtures

| Config | Purpose |
|---|---|
| `tests/tiny_semseg` | one-epoch semantic-segmentation recipe used by the first experiment |
| `tests/tiny_semseg_crash` | deterministic crash/resume test recipe |

Use `tiny_semseg` for the {doc}`quickstart <../getting_started/quickstart>`;
neither fixture is a scientific benchmark.

### Panda / Sonata pretraining

| Config | Purpose |
|---|---|
| `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` | Sonata with PT-v3m2 on PILArNet-M |
| `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8` | Sonata with the PT-v3m8 backbone variant |
| `panda/pretrain/pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable-parquet` | local/Hugging Face Parquet reader variant; **currently unresolved** |
| `panda/pretrain/pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable-parquet-stream` | iterable Parquet reader variant; **currently unresolved** |

The two Parquet configs inherit
`pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable.py`, which is
not committed at this revision.
{py:meth}`~pimm.utils.config.Config.fromfile` raises `FileNotFoundError`.
Treat them as incomplete provenance records until the base config is restored
or they are made self-contained.

### Panda semantic segmentation

| Config | Adaptation |
|---|---|
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` | full fine-tune |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-dec` | frozen encoder / decoder adaptation |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin` | frozen backbone / task-head training |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-scratch` | full config without a warm-start supplied at launch |

### Panda Detector

All paths below are under `panda/panseg/`.

| Model | Target | Variants |
|---|---|---|
| `detector-v1m1-pt-v3m2-ft-pid` | particle/PID | `-dec`, `-fft`, `-scratch` |
| `detector-v1m1-pt-v3m2-ft-vtx` | interaction/vertex | `-dec`, `-fft`, `-scratch` |
| `detector-v1m2-pt-v3m2-ft-pid` | particle/PID with v1m2 detector | `-dec`, `-fft`, `-scratch` |
| `detector-v4-pt-v3m2-ft-pid` | configurable v4 particle/PID detector | `-dec`, `-fft`, `-scratch` |
| `detector-v4-pt-v3m2-ft-vtx` | configurable v4 interaction/vertex detector | `-dec`, `-fft`, `-scratch` |

The table expands to all 15 committed files. There is no v1m2 `vtx` config in
this checkout. Start new work from v4 unless a checkpoint requires the older
registered structure, and verify the target field/class order in the file.

### PoLAr-MAE

| Config | Purpose |
|---|---|
| `polarmae/pretrain-polarmae-pilarnet` | masked-autoencoder pretraining |
| `polarmae/semseg/semseg-polarmae-pilarnet-fft` | full semantic fine-tuning |
| `polarmae/semseg/semseg-polarmae-pilarnet-peft` | frozen-encoder/head-oriented parameter-efficient recipe |
| `polarmae/semseg/semseg-polarmae-pilarnet-fft-reproduce` | evaluate the released fine-tuned checkpoint |

### Other pretraining research

| Config | Purpose |
|---|---|
| `hmae/pretrain-hmae-v1m1-pilarnet-1m-amp-seed0` | HMAE v1m1 with the configured hierarchical backbone |
| `hmae/pretrain-hmae-v1m1-ptv4-pilarnet-1m-amp-seed0` | HMAE v1m1 with the PTv4 variant in the file |
| `lejepa/pretrain/pretrain-lejepa-v1m5-pilarnet` | LeJEPA v1m5 research recipe |
| `lejepa/pretrain/pretrain-lejepa-v1m5-pilarnet-small` | smaller-data LeJEPA v1m5 variant |
| `voltmae/pretrain-voltmae-pilarnet` | Volt-MAE v1m1 pretraining |
| `voltmae/pretrain-voltmae-pilarnet-L` | larger Volt-MAE v1m1 variant |
| `voltmae/pretrain-voltmae-v1m2-pilarnet` | point-set Volt-MAE v1m2 baseline |

### Other detector data

| Config | Purpose |
|---|---|
| `detector/semseg/semseg-pt-v3m2-jaxtpc-5cls` | PT-v3m2 semantic segmentation on JAXTPC; requires `JAXTPC_DATA_ROOT` or an explicit root |

## Shared bases

| Path | Role |
|---|---|
| `configs/_base_/default_runtime.py` | trainer/tester, hooks, global batch/worker defaults, seed, AMP and runtime settings |
| `configs/detector/_base_/jaxtpc_seg.py` | JAXTPC data root, class names, and split definitions |
| `configs/_base_/dataset/scannetpp.py` | inherited Pointcept dataset base retained in the repository; not a top-level pimm recipe |

Do not launch a `_base_` file as if it were a complete study.

## Launch profiles

| Profile | Executor environment |
|---|---|
| `local` | current node, no container; GPU count defaults to `auto` |
| `slurm` | generic Slurm base inherited by cluster profiles |
| `s3df` | S3DF bare-metal environment |
| `s3df-container` | S3DF Singularity environment |
| `nersc` | NERSC bare-metal environment prepared by `scripts/nersc_env.sh` |
| `nersc-container` | NERSC Shifter environment |

Site files contain real project accounts, partitions, modules, mounts, and
network settings. Fork or override them for your allocation; their presence is
not authorization to use those accounts.

## Adding a recipe

Create a small child config, keep storage paths external, and add a bounded
validation command. Then follow {doc}`Add a model <../extend/add_model>` or
{doc}`Add a dataset <../extend/add_dataset>` as appropriate. A committed recipe
should declare its data revision, target/class contract, checkpoint provenance,
and status rather than relying on its filename.
