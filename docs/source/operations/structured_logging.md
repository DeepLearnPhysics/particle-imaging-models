# Structured logging

pimm's structured logger writes a chronological, machine-readable trace of events taking
place on each training process (aka GPU). It is intended for diagnosing issues with distributed
training, like finding the rank, step, and phase where a run became slow, stalled, restarted,
or failed. It is essentially a copy of [torchtitan](https://github.com/pytorch/torchtitan)'s
own structured logger, which you can learn about [here](https://github.com/pytorch/torchtitan/tree/main/torchtitan/observability/structured_logger).

Structured logging complements the normal training log, W&B or
TensorBoard metrics, and `torch.profiler` -- it's not a replacement! And it must be enabled explicitly.

Design principles:

* LLM friendly: Emit structured, per-rank data that can be queried during and after a run.
* Stay invisible to users -- no metric dictionaries to pass around.
* Fail at startup when requested tracing is incomplete or cannot initialize;
  never silently run without observability that the user explicitly enabled.
* Support pluggable backends via handler factories.


## Quickstart

The default runtime configuration contains:

```python
structured_logging = dict(
    enabled=False,
    trace_hooks=False,
    batch_stats_every=1,
    max_file_size_mb=128,
    backup_count=3,
)
```

You can enable it for a one-off local run with a simple training config override:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- structured_logging.enabled=true
```

Or, for a reusable experiment variant, set it explicitly in a Python config:

```python
...
structured_logging = dict(
    enabled=True,
    max_file_size_mb=256,
    backup_count=3,
)
```

The options mean:

- `enabled`: create structured trace files and instrument the training
  lifecycle. Calls to the extension API are no-ops when false.
- `trace_hooks`: time every individual hook invocation.
  Leave this false for normal diagnosis because per-hook events can be noisy.
- `batch_stats_every`: emit rank-local batch-size statistics every N steps;
  set `0` to disable them.
  Set it to zero to disable those scalar records.
- `max_file_size_mb`: rotate a rank's active JSONL file at approximately this
  size.
- `backup_count`: retain this many rotated segments in addition to the active
  file. The resulting bound is approximately
  `(backup_count + 1) * max_file_size_mb` per rank.

Initialization is intentionally strict. A missing setting, an unwritable output
directory, or a broken custom handler raises during startup instead of silently
continuing without the requested trace.

## What is recorded

The trainer records process and build lifecycle events, followed by nested
training phases such as:

```text
step
  data_fetch
  hooks.before_step
  batch_to_device
  forward
  backward
  optimizer
  hooks.after_step
```

Each record, saved in a set of `.jsonl` files (one for each rank/GPU), includes
the context available at emission time:

- absolute and restart-relative training step;
- epoch and iteration within the epoch;
- global rank, local rank, and world size;
- host and process identifiers;
- Submitit attempt and Slurm job identifiers when present;
- trace ID, source, timestamp, caller, and sequence ID;
- optional step tags and scalar values.

## Output layout

Files are written alongside the other experiment artifacts:

```text
<save_path>/
  config.py
  resolved_config.json
  train.log
  structured_logs/
    training.global_rank_0.<timestamp>-<random>.jsonl
    training.global_rank_1.<timestamp>-<random>.jsonl
    ...
```

### Custom handlers

Advanced users can replace the JSONL backend by setting
`PIMM_STRUCT_LOGGER_HANDLERS` to a comma-separated list of importable factory
paths:

```bash
export PIMM_STRUCT_LOGGER_HANDLERS=my_package.trace.register_handler
```


A handler factory takes the args `sl.init_structured_logger()` forwards and
attaches one handler to `structured_logger`. Example: stream events to a remote
database instead of writing to disk, and enrich each record with cluster metadata
along the way.

```python

import logging
import os

from pimm.observability.structured_logger.jsonl_handler import (
    TraceJsonlFormatter,
)
from pimm.observability.structured_logger.structured_logging import (
    TraceEventsOnlyFilter,
)


class MyDBFormatter(TraceJsonlFormatter):
    """Enrich each record with backend-specific fields before serialization."""

    def _log_dict(self, record):
        d = super()._log_dict(record)
        d["cluster"] = os.environ.get("CLUSTER_NAME", "unknown")
        return d


class MyDBHandler(logging.Handler):
    """Send each trace event to a remote DB as it's emitted."""

    def __init__(
        self,
        rank: int,
        source: str,
        db_url: str,
        *,
        world_size: int,
        attempt: int,
        job_id: str | None,
        trace_id: str | None,
    ):
        super().__init__()
        self.client = MyDBClient(db_url)
        self.setFormatter(
            MyDBFormatter(
                rank=rank,
                source=source,
                world_size=world_size,
                attempt=attempt,
                job_id=job_id,
                trace_id=trace_id,
            )
        )
        self.addFilter(TraceEventsOnlyFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.client.insert_row(self.format(record))
        except Exception:
            self.handleError(record)


def register_my_db_handler(
    *,
    structured_logger: logging.Logger,
    rank: int,
    source: str,
    world_size: int,
    attempt: int,
    job_id: str | None,
    trace_id: str | None,
    **kw,
) -> None:
    structured_logger.addHandler(
        MyDBHandler(
            rank=rank,
            source=source,
            db_url="mydb://...",
            world_size=world_size,
            attempt=attempt,
            job_id=job_id,
            trace_id=trace_id,
        ),
    )
```


## Inspect a run

Summarize a run directory:

```bash
pimm trace summarize exp/panda/pretrain/<run-name>
```

The command accepts either the run directory or its `structured_logs/`
directory. It reports:

- every process's final step, relative step, event, and timestamp;
- unclosed spans and their last known phase;
- phase count, p50, p95, and maximum duration;
- the slowest rank, attempt, and step for each phase.

This is normally the first command to use after a hang or timeout. For example,
seven ranks ending at `optimizer_end` while one rank ends at
`data_fetch_start` immediately narrows the investigation.

Export a Chrome Trace document:

```bash
pimm trace export exp/panda/pretrain/<run-name>
```

The default output is:

```text
<save_path>/analysis/structured_trace.json
```

Choose another destination with:

```bash
pimm trace export exp/panda/pretrain/<run-name> \
  --output /tmp/pimm-trace.json
```

Open the resulting file in [Perfetto](https://ui.perfetto.dev/) or
`chrome://tracing`. Nested spans in one task share a track, truly concurrent
tasks receive separate tracks, and sequential tasks reuse tracks. If a process
ended after writing a `_start` record but before its matching `_end`, export
keeps the phase as an incomplete span extending to that process's last
timestamp. It is marked with `incomplete: true` instead of silently
disappearing.

Readers include rotated `*.jsonl*` segments. A malformed or truncated final
line produces a warning and is skipped, allowing the rest of a crash trace to
remain usable.

## Instrument an extension

Models, hooks, and other main-process research extensions can use the public API
without checking whether tracing is enabled:

```python
from pimm.observability.structured_logger import (
    add_step_tag,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
)


with log_trace_span("model.decoder"):
    output = self.decoder(features)

log_trace_scalar({"batch.local_points": int(input_dict["coord"].shape[0])})

add_step_tag("custom_evaluation")
log_trace_instant("custom_evaluation_complete")
```

`log_trace_span()` works as either a context manager or a decorator, including
on async functions:

```python
@log_trace_span("remote_rollout")
async def rollout(request):
    return await actor.generate(request)
```

Prefer stable, low-cardinality phase names. Put changing numeric information in
`log_trace_scalar()` rather than embedding step IDs or sizes into the phase
name. Avoid logging tensors directly; convert only values whose synchronization
cost is already acceptable.

The logger is initialized and shut down by the training entry point. Extension
code should not call `init_structured_logger()` itself. Instrument the trainer's
main-process `data_fetch` boundary rather than dataset worker processes.

## Timing and storage considerations

Span durations are host-visible wall time. CUDA work is asynchronous, so a
`forward` or `optimizer` span can include enqueue time rather than the precise
duration of every GPU kernel. The structured trace intentionally does not call
`torch.cuda.synchronize()`, because synchronization would perturb training.
Use `torch.profiler` when accurate CPU/CUDA operator and kernel timing is
required.

Wall-clock timestamps are used to align records from different ranks. Comparing
absolute timestamps across nodes assumes the hosts' clocks are synchronized.
Durations measured within one process do not have that dependency.

JSONL emission is synchronous and flushed so the final pre-crash event is
likely to survive. That makes the trace diagnostically useful, but it is not
free. Event size depends on metadata, and detailed per-step tracing can generate
hundreds of bytes per event. At scale, many events per step across many ranks
can consume gigabytes and add shared-filesystem traffic.

Use bounded rotation for incident diagnosis, leave `trace_hooks` disabled unless
needed, and avoid high-frequency custom scalar events. For full-run performance
analysis, provision storage intentionally and validate overhead on the target
filesystem.

W&B and TensorBoard remain the right destinations for aggregated experiment
metrics and comparisons. The structured logger retains rank-local ordering and
phase boundaries that aggregate metric writers intentionally discard.
`torch.profiler` provides operator-level CPU/CUDA detail for selected windows;
the structured logger provides a lightweight, whole-process breadcrumb trail
for deciding which rank, step, or phase deserves profiling.
