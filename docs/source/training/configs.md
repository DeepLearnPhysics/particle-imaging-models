# Configs and overrides

An experiment config is a Python file that can inherit from others; a launch config is YAML plus flags. Both have a fixed precedence.

## Experiment configs

```python
# configs/my_study/semseg.py
_base_ = ["../panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

seed = 7
batch_size = 16
optimizer = dict(lr=3e-5)
```

`_base_` paths are relative to the file. Dicts are merged recursively with the base, while scalars and lists are replaced; if you redefine `hooks`, `transform`, `criteria` or `param_dicts`, you drop the inherited list.

## Overrides on the command line

Launcher flags come before a bare `--`; experiment overrides come after it as `key=value` tokens:

```bash
uv run pimm launch --train.config my_study/semseg --resources.nproc-per-node 4 --run.name semseg-seed7 \
  -- batch_size=16 optimizer.lr=3e-5 data.train.max_len=100000
```

Use dotted keys to reach nested dicts; values are parsed like YAML. Tokens after `--` that start with `--` are rejected.

## Precedence

```text
experiment:  base config(s)  →  child config  →  overrides after --
launch:      launch/defaults.yaml  →  launch/sites/<site>.yaml  →  --recipe YAML  →  flags
```

You can read the result in `config.py` and `resolved_config.json` in the run directory.

## Common experiment fields

Defaults come from `configs/_base_/default_runtime.py`; many of them are overridden in recipes.

| Field | Default | Meaning |
|---|---|---|
| `weight` | `None` | weights to load at start: a path or `hf://org/repo` |
| `resume` | `False` | restore trainer state as well as weights |
| `evaluate` | `True` | run validation during training |
| `seed` | `None` | random seed; drawn and recorded when `None` |
| `batch_size` | `16` | events per step, across all GPUs |
| `batch_size_val`, `batch_size_test` | `None` | across all GPUs; `None` means one event per GPU |
| `num_worker` | `16` | data-loader workers, across all GPUs |
| `epoch` | `800` | total passes over the training data |
| `eval_epoch` | `800` | rounds the training is split into; validation and checkpoints follow each round |
| `clip_grad` | `None` | gradient-norm clipping threshold |
| `enable_amp`, `amp_dtype` | `False`, `"bfloat16"` | mixed precision |
| `param_dicts` | `None` | learning-rate scales for parameter groups, for example `[dict(keyword="block", lr_scale=0.1)]` |
| `hooks` | see [How pimm works](../get-started/how-pimm-works.md#the-training-loop-and-hooks) | ordered lifecycle hooks |
| `train`, `test` | `DefaultTrainer`, `SemSegTester` | trainer and tester |
| `structured_logging` | disabled | per-rank traces; see [Monitor and debug](monitor.md#structured-traces) |
| `use_wandb` | set in recipes | log to W&B when `True`, TensorBoard otherwise |
| `checkpoint_format` | `"standard"` | `"standard"` (split) or `"legacy"` (single file) |
| `parallel` | DDP | `dict(strategy="ddp")`, or the experimental `"fsdp2"` |

For launcher flags, see [Command line](../reference/cli.md).

## Check before running

Add `--dry-run` to resolve both configs and print what would run, without building the model or opening data:

```bash
uv run pimm launch --train.config my_study/semseg --dry-run
```
