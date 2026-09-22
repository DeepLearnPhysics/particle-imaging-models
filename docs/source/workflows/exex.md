# Experimental exex execution

Exex is a required dependency; its execution adapter is opt-in. Set
`PIMM_USE_EXEX=1` with the existing pimm CLI. Unset it or set it to `0` to keep
the existing launcher. There is no separate launcher script or checkpoint hook
to configure, and failures never trigger silent fallback.

## Run

Install the locked dependencies with `uv sync`. The author and execution
runtimes, including containers, need the pinned exex dependency.

```sh
# Inspect the plan without staging source or submitting a job.
PIMM_USE_EXEX=1 uv run pimm submit --site s3df \
  --train.config tests/tiny_semseg --resources.nproc-per-node 1 --dry-run

# Run locally using the same config interface.
PIMM_USE_EXEX=1 uv run pimm launch \
  --train.config tests/tiny_semseg --resources.nproc-per-node 1

# Two Slurm nodes, with two GPU workers per node.
PIMM_USE_EXEX=1 uv run pimm submit --site s3df \
  --train.config tests/tiny_semseg --resources.nnodes 2 --resources.nproc-per-node 2 \
  -- batch_size_val=4 batch_size_test=4

# NERSC interactive allocations, with up to three total attempts.
PIMM_USE_EXEX=1 uv run pimm submit --site nersc --interactive \
  --train.config tests/tiny_semseg --resources.nproc-per-node 1 \
  --resources.qos interactive --chain.jobs 3
```

Use your training config or existing `--recipe`; datasets must be available on
the execution site. Site, resource, container, environment and setup settings
select the target. There is no placement resolver or second site file.

## Source and runtime

Exex captures the Git-selected `pimm/` and `configs/` files, including
uncommitted changes, and stages the frozen source to the selected host. It does
not rebuild dependencies or synchronize a mutable remote checkout.
`paths.repo_root` can identify a prepared execution runtime; source always comes
from the author's checkout. Dependencies outside those source directories must
already be installed in the runtime.

The old `train.code_copy` and `container.repo_mount` choices are superseded by
frozen-source execution. Do not bind a mutable checkout over the captured source.

## Continuation

Pimm runs torchrun and automatically restores its checkpoint and saves at safe
pause boundaries. Exex retains the completed checkpoint and handles the next
allocation. `chain.jobs` includes the initial attempt;
`resources.signal_delay_s` reserves time to reach a safe point and capture state.
Crashes and failed checkpoints are not retried.

Each attempt has its own output directory. The named `checkpoint` artifact holds
that attempt's training outputs and resumable model state. The previous artifact
supplies the next attempt's input; training writes
to the new output directory. An initial `--train.resume` requires `--train.weight`
pointing to a complete checkpoint visible to the worker.

Execution continuation does not select W&B history. Set `PIMM_WANDB_HISTORY`
independently of `PIMM_USE_EXEX`:

- `new` (default): create a new tracking run, even when training resumes.
- `append`: continue the checkpoint's tracking run without deleting history.
- `fork`: create a child inheriting history through the checkpoint; requires
  W&B permission. Permission failures are not silently downgraded.

`append` and `fork` require checkpointed tracking identity and online mode.
Rewind is unsupported. Explicit training config `wandb_history` takes precedence
over the environment variable. Exex owns tracking identity/history and links;
pimm keeps metric buffering and the semantic `train/global_step` axis.
Keep credentials in runtime credential files or the worker environment, not
recorded arguments or launch `env` dictionaries.

## Storage and remote submission

The author catalog is `.exex/` in the checkout. Target staging is selected by
`submit.folder`, defaulting to `paths.exp_root/.exex`. These directories contain
retained execution state, not source files; do not commit or casually delete them.
The launcher prints the experiment ID. From the author checkout, inspect it
with `exex experiments`, `exex status ID`, or `exex logs ID 1`.

For S3DF → NERSC, set `--submit.host nersc` and an absolute target
`--paths.exp-root`. SSH uses your existing configuration. Select an absolute
target interpreter for a host runtime, or the prepared Shifter image/interpreter.
Shifter clears inherited environment; use explicit `env` and `setup` settings.

Multi-node Slurm launches one torchrun coordinator per node using srun outside
the container. Staging must be shared by all nodes; continuation keeps node/GPU
counts and batch settings fixed. An attached `salloc` launcher must stay alive.

The current adapter supports single-node Local and multi-node Slurm, including
multiple GPUs per node. Cloud execution, array continuation, watchdogs and
automatic cross-site placement are not supported.
