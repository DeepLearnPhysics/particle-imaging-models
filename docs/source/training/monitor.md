# Monitor and debug

Every run has a `train.log`. Metrics are logged to Weights & Biases or TensorBoard; per-GPU traces and profiling are opt-in.

## Weights & Biases

```bash
uv run wandb login
uv run pimm launch --train.config <recipe> --run.wandb-project <project> --run.wandb-name <name> -- use_wandb=True
```

By default, the project is `pimm` and the display name is the run name. Only the first process (rank 0) has a W&B run, so a multi-GPU job is one W&B run. In W&B, you get the resolved experiment config and these metrics:

| Key | Content |
|---|---|
| `train_batch/*` | scalar model outputs, every step |
| `train/*` | epoch means of the same |
| `val/*` | evaluator metrics |
| `params/lr` | learning rate |

`train_batch/*` values come from rank 0 unless you reduce them across GPUs in your model. Checkpoints, code and launch files stay in the run directory and are not uploaded. The W&B run is recorded in the checkpoints, so a resumed run continues in the same W&B run.

## TensorBoard

Set `use_wandb=False` to write the same keys to TensorBoard event files in the run directory. Open them with `tensorboard --logdir exp`.

## Diagnostic hooks

Add these hooks to `hooks` when you need them:

| Hook | Logged |
|---|---|
| `ParameterCounter` | total, frozen and trainable parameters at startup |
| `GradientNormLogger` | global and per-layer gradient norms |
| `ResourceUtilizationLogger` | CPU, RAM, GPU memory and utilization |
| `FeatureStdMonitor` | spread of student and teacher features |
| `PrototypeUsageLogger` | prototype occupancy and entropy |
| `LogitEntropyLogger` | entropy of classification logits |
| `RuntimeProfiler` | a `torch.profiler` trace of selected steps |

```python
hooks = [
    dict(type="ParameterCounter", show_details=True),
    dict(type="ResourceUtilizationLogger", log_frequency=10, log_per_gpu=True, per_process=True),
    dict(type="GradientNormLogger", log_frequency=10),
    # then the evaluator and checkpoint hooks
]
```

The data time, batch time and remaining-time estimate in `train.log` come from `IterationTimer`. When the GPU is idle waiting for data, data time is high compared with batch time.

## Structured traces

Structured traces are a record of what each GPU process was doing, step by step, in one JSON-lines file per process. Use them to find the GPU, step and phase where a distributed run slowed down, stalled or failed. Turn them on for a run:

```bash
uv run pimm launch --train.config <recipe> -- structured_logging.enabled=true
```

The files appear in `<run>/structured_logs/`. Summarize them, or merge them into a timeline:

```bash
uv run pimm trace summarize exp/<group>/<run>
uv run pimm trace export exp/<group>/<run>
```

Run `summarize` to print each process's last step and phase, open spans, and phase durations (median, 95th percentile, maximum). Run `export` to write `<run>/analysis/structured_trace.json`, which you can open in [Perfetto](https://ui.perfetto.dev/).

| Option | Default | Effect |
|---|---|---|
| `enabled` | `False` | write traces |
| `trace_hooks` | `False` | time every hook call |
| `batch_stats_every` | `1` | batch-size records every N steps, or none with `0` |
| `max_file_size_mb` | `128` | rotate each process's file at this size |
| `backup_count` | `3` | rotated files kept per process |

Span times are host wall-clock times. CUDA execution is asynchronous, so a span can end before its GPU work does; use `RuntimeProfiler` for kernel timing. In your own code, you can add records with `log_trace_span`, `log_trace_scalar` and `log_trace_instant` from `pimm.observability.structured_logger`; these calls are no-ops when tracing is off.
