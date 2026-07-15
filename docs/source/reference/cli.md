# Command-line reference

pimm exposes three commands. Training-config values are intentionally not
typed launcher flags: put them after a bare `--` as `key=value` tokens.

| Command | Use it for | Does not do |
|---|---|---|
| `pimm launch` | run on the current node or inside an existing allocation | submit a scheduler job |
| `pimm submit` | submit through Slurm with Submitit, or request an interactive allocation | evaluate a completed experiment |
| `pimm export` | consolidate model weights into a portable directory and optionally upload it | save resumable trainer state |

Run commands through the checkout's locked environment:

```bash
uv run pimm --help
uv run pimm launch --help
uv run pimm submit --help
uv run pimm export --help
```

## The override boundary

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  --run.name smoke \
  -- \
  epoch=1 \
  batch_size=4 \
  data.train.max_len=32
```

Everything before `--` configures execution. Everything after it patches the
Python experiment config. Post-separator tokens must contain `=` and must not
start with `--`; values are parsed with YAML scalar/list/map syntax. Dotted
keys patch nested mappings.

## `pimm launch`

```text
pimm launch [launcher options] -- [EXPERIMENT.KEY=VALUE ...]
```

`launch` always selects the local executor, even if a different `--executor`
value is passed. “Local” means run in the current shell context: this can be a
workstation, one compute node, or an allocation obtained separately.

```bash
# Resolve without starting training
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 4 \
  --dry-run
```

The `local` site defaults `resources.nproc_per_node` to `auto`, which asks
`scripts/train.sh` to use all visible GPUs. Set an integer for a predictable
world size.

## `pimm submit`

```text
pimm submit [launcher and Slurm options] -- [EXPERIMENT.KEY=VALUE ...]
```

Always pass `--site` explicitly in scripts. The parser currently defaults
`submit` to the repository's `s3df` profile, which is convenient for that site
but not portable.

```bash
uv run pimm submit \
  --site nersc \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --dry-run
```

`--interactive` renders and runs `salloc ... srun ...` in the terminal instead
of queueing a Submitit batch job. Interactive submission is single-shot and
rejects `--chain.jobs` greater than one.

If `--submit.host HOST` is set, pimm SSHes to that host and invokes `pimm
submit` there. `--no-remote` prevents that second hop; pimm adds it internally
to avoid recursion.

## Launcher option groups

Hyphens in flags map to underscores in the launch dataclasses. Boolean flags
have positive and negative forms, for example `--run.timestamp` and
`--run.no-timestamp`.

| Group | Flags users commonly set |
|---|---|
| selection/output | `--site`, `--recipe`, `--dry-run`, `--output` |
| paths | `--paths.repo-root`, `--paths.exp-root` |
| resources | `--resources.nnodes`, `--resources.nproc-per-node`, `--resources.cpus-per-proc`, `--resources.time`, `--resources.mem` |
| run identity | `--run.name`, `--run.[no-]timestamp`, `--run.wandb-name`, `--run.wandb-project`, `--run.wandb-api-key` |
| training handoff | `--train.config`, `--train.weight`, `--train.[no-]resume`, `--train.[no-]code-copy`, `--train.python` |
| Slurm | `--slurm.account`, `--slurm.partition`, `--slurm.qos`, `--slurm.constraint`, `--slurm.dependency`, `--slurm.output`, `--slurm.error`, `--slurm.gpu-directive`, `--slurm.job-name`, `--slurm.signal-delay-s` |
| container | `--container.runtime`, `--container.image`, `--container.module`, `--container.binds`, `--container.repo-mount`, `--container.unset-env`, `--container.setup`, `--container.interpreter` |
| requeue | `--chain.jobs`, `--chain.[no-]resume-first`, `--chain.wandb-group`, `--chain.wandb-job-type` |
| rendezvous | `--rdzv.endpoint`, `--rdzv.id`, `--rdzv.backend` |
| remote submit | `--submit.host`, `--submit.folder`, `--submit.setup`, `--no-remote` |

`--recipe` is a YAML path relative to the repository root, for example
`--recipe launch/runs/e050_tail.yaml`. `--output PATH` writes the redacted
local script, Submitit manifest, or interactive `salloc` command. It does not
redirect training logs.

Three advanced mappings are deliberately absent from Tyro's flag list:
`env`, `train.options`, and `slurm.additional_parameters`. Put them in launch
YAML. Prefer the post-`--` tail over `train.options` for one-off experiment
overrides.

:::{caution}
`--executor` is visible for schema compatibility, but the command verb wins:
`launch` forces local and `submit` forces batch or interactive Slurm. Choose the
verb rather than setting this field.
:::

## `pimm export`

```text
pimm export [checkpoint] [output_dir] [options]
```

With `--run-dir`, the checkpoint positional argument defaults to `last` and is
resolved under `<run-dir>/model/`. The output directory is required unless
`--dry-run` is set.

```bash
uv run pimm export \
  --run-dir exp/panda/semseg/my-run \
  last \
  artifacts/panda-semantic
```

| Option | Meaning |
|---|---|
| `--run-dir DIR` | infer checkpoint names and the run config from an experiment directory |
| `--config FILE` | provide a config when it cannot be inferred |
| `--model-card FILE` | copy Markdown into the export as `README.md` |
| `--device DEVICE` | device used while consolidating tensors; default `cpu` |
| `--no-safe-serialization` | write `model.bin` instead of `model.safetensors` |
| `--push-to-hub ORG/REPO` | upload the completed export |
| `--public` | create a public Hub repository; default is private |
| `--token TOKEN` | explicit Hub token; prefer `hf auth login` or `HF_TOKEN` |
| `--dry-run` | print resolved paths and exit without writing |

See {doc}`Export and publication <../models/export>` for artifact semantics and a
strict round-trip check.

## Evaluation is currently a script

There is no `pimm evaluate` subcommand. The supported standalone path is:

```bash
uv run sh scripts/test.sh \
  -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  -n <experiment-name> \
  -w model_best
```

The script accepts `-p`, `-c`, `-n`, `-w`, `-g`, and `-m`, but its current
execution path is single-process. Its `-c` value selects a config from the
current checkout, not the saved run's `config.py`. Read {doc}`Evaluate an
experiment <../workflows/evaluate>` for the exact saved-config command before
reporting metrics.

## Exit-before-spend checks

Use these before a job with data or GPUs:

```bash
uv run pimm launch --train.config <config> --dry-run
uv run pimm submit --site <site> --train.config <config> --dry-run
uv run pimm export --run-dir <run> --dry-run
```

Dry runs validate cheap path/config conditions and render the command. They do
not construct the dataset or model, test credentials, inspect scheduler state,
or prove that a container/data mount is reachable on compute nodes.
