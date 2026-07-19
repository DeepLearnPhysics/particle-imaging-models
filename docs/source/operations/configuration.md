# Configuration

pimm separates experiment state from execution state. This distinction is the
key to moving one recipe across machines without copying cluster details into
the science config.

## Two layers

| Layer | Format | Examples |
|---|---|---|
| experiment | Python under `configs/` | model, data, transforms, loss, optimizer, scheduler, hooks, epochs, global batch sizes |
| execution | YAML under `launch/` plus launcher flags | site, nodes/GPUs/CPUs, paths, Slurm, container, environment, run name, resume |

## Python inheritance

```python
# configs/my_study/semseg.py
_base_ = ["../panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

seed = 7
batch_size = 16
data = dict(train=dict(max_len=100_000))
optimizer = dict(lr=3e-5)
```

Paths in `_base_` are relative to the child file. Dictionaries merge
recursively. Scalars replace scalars. **Lists replace lists**; redefining
`hooks`, `transform`, `criteria`, or `param_dicts` drops the inherited list.

## Overrides from the launcher

Launcher flags come before a bare `--`; experiment overrides come after it:

```bash
uv run pimm launch \
  --train.config my_study/semseg \
  --resources.nproc-per-node 4 \
  --run.name semseg-seed7 \
  -- \
  batch_size=16 \
  optimizer.lr=3e-5 \
  data.train.max_len=100000
```

Values are parsed as YAML-like scalars/containers and dotted keys patch nested
dictionaries. Tokens beginning with `--` after the separator are invalid.

Use an override for a short probe. Use a child config for anything reviewed,
resumed, compared, or published.

## Precedence

Experiment values:

```text
base Python config(s) → child Python config → post-`--` overrides
```

Execution values:

```text
launch/defaults.yaml → site YAML → optional recipe YAML → launcher flags
```

`pimm launch`/`submit` materialize both layers into the run. `config.py` and
`resolved_config.json` are the authoritative experiment record after
inheritance and overrides.

## Common experiment fields

| Field | Meaning |
|---|---|
| `weight` | model/checkpoint source for warm start or resume |
| `resume` | restore trainer state rather than only weights |
| `seed`, `deterministic` | RNG initialization and deterministic-mode request |
| `batch_size` | global training event count across all ranks |
| `batch_size_val`, `batch_size_test` | global explicit totals; `None` means one event per rank |
| `num_worker` | global worker total across all ranks |
| `epoch`, `eval_epoch` | loop/evaluation scheduling fields used by trainers |
| `enable_amp`, `amp_dtype` | automatic mixed precision behavior |
| `model` | registered top-level model and nested components |
| `data` | dataset/split configs and scientific metadata |
| `optimizer`, `scheduler`, `param_dicts` | optimization |
| `hooks` | ordered lifecycle behavior |
| `train`, `test` | trainer and tester implementations |
| `checkpoint_format` | `standard` or `legacy` |
| `parallel` | DDP/FSDP2 strategy settings |

Constructor-level details belong in the {doc}`API <../api/index>`; do not
duplicate entire live configs into prose.

## Common execution fields

| Flag/YAML path | Meaning |
|---|---|
| `site` | profile under `launch/sites/` |
| `paths.repo_root`, `paths.exp_root` | checkout and experiment roots |
| `resources.scheduler` | `local` or `slurm` |
| `resources.nnodes`, `nproc_per_node`, `cpus_per_proc` | nodes, GPUs/processes per node, and CPUs per process |
| `resources.time`, `mem`, `account`, `partition`, `qos`, `constraint`, `output`, `gpu_directive` | scheduler requests and policy |
| `run.name`, `run.timestamp` | experiment path naming |
| `train.config`, `train.weight`, `train.resume`, `train.code_copy` | training entry and provenance behavior |
| `container.*` | runtime, image, binds, interpreter |
| `setup` | shell bootstrap lines run before training |
| `chain.*` | repeated/requeued job attempts |

The current exhaustive flag set comes from the typed dataclasses and is listed
in {doc}`CLI reference <../reference/cli>`.

## Inspect before running

```bash
uv run pimm launch --train.config <config> --dry-run
```

After a bounded run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

run = Path("exp/<group>/<run>")
cfg = json.loads((run / "resolved_config.json").read_text())
for key in ("seed", "batch_size", "num_worker", "enable_amp", "checkpoint_format"):
    print(key, cfg.get(key))
print("model", cfg["model"]["type"])
print("train data", cfg["data"]["train"])
PY
```

## Common mistakes

- Writing `batch_size` as per-GPU. It is global and must divide world size.
- Editing a base config for one study. Create a child so other recipes do not
  change.
- Redefining one hook in a list and accidentally deleting every inherited hook.
- Passing `--options` after the launcher's bare `--`; use bare `key=value`.
- Keeping absolute data/checkpoint paths in a config intended for publication.
- Assuming resume re-reads the current original config rather than the saved run
  artifacts and launch state.

## Next

- {doc}`CLI <../reference/cli>`.
- {doc}`Checkpoints and resume <checkpoints>`.
- {doc}`Slurm execution <../workflows/slurm>`.
