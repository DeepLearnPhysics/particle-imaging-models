<!--
from: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/README.md#L1-L189
-->

# pimm observability

Structured logging for distributed training. It emits per-rank JSONL events for
phase timing, diagnostics, and post-hoc analysis.

Design principles:

- Emit structured, per-rank data that can be queried during and after a run.
- Work before `torch.distributed` initialization by using the environment set
  by `torchrun`.
- Stay invisible to model and hook APIs--there are no metric dictionaries to
  pass around.
- Keep tracing optional and support pluggable backends through handler
  factories.

## Quickstart

```python
from pimm.observability import structured_logger as sl


# Register handlers that save logs to local JSONL, a database, or another
# backend. Call this once per process before any trace calls.
sl.init_structured_logger(source="training", output_dir="./outputs")
sl.log_trace_instant("training_start")

loaded_step = 0
for step in range(loaded_step + 1, num_steps + 1):
    # Stamp every subsequent record with `step` and `relative_step`.
    sl.set_step(step, relative_step=step - loaded_step)

    if should_garbage_collect:
        # Appends "gc" to `step_tags` on every record for this step; tags
        # reset at the next set_step() call.
        sl.add_step_tag("gc")
        with sl.log_trace_span("gc_collect"):
            run_gc()

    with sl.log_trace_span("fwd_bwd"):
        output = model(batch)
        loss.backward()

    with sl.log_trace_span("optimizer"):
        optimizer.step()

    sl.log_trace_scalar(
        {
            "num_trainable_tokens": num_trainable_tokens,
            "batch_size": bsz,
        }
    )

sl.shutdown_structured_logger()
```

pimm initializes this logger from the training entrypoint. Enable it for a run
with:

```bash
pimm launch \
  --train.config <config> \
  -- structured_logging.enabled=true
```

## API reference

See docstrings for the complete argument reference:

- `sl.init_structured_logger(source, output_dir, rank=None, enable=True, ...)`
  wires up handlers. Call it once per process before any trace call. Passing
  `enable=False` makes all trace calls no-ops. Configuration and handler
  initialization errors propagate to the caller.
- `sl.shutdown_structured_logger()` flushes and closes registered handlers.
- `sl.log_trace_span(event_type, description=None, *, stacklevel=2)` is a
  context manager/decorator that emits `_start`, `_end`, and optional `_error`
  records.
- `sl.log_trace_instant(event_type, *, stacklevel=2)` emits a point-in-time
  marker.
- `sl.log_trace_scalar(scalars, *, stacklevel=2)` emits `metric_value` records
  from a `{name: number}` dictionary.
- `sl.set_step(step, *, relative_step=None, epoch=None, iteration=None)` stamps
  subsequent records with training position and clears the previous step's
  tags.
- `sl.add_step_tag(tag)` and `sl.clear_step_tags()` annotate the current step.
- `PIMM_STRUCT_LOGGER_HANDLERS` selects handler factories through the
  environment.

## Flow of information

When user code calls one of the `sl.*` helpers:

```text
user code
    │   with sl.log_trace_span("fwd_bwd"):
    │       ...
    │
    │   # On entry, log_trace_span calls:
    │       _structured_logger.info(
    │           msg="[step 5] fwd_bwd_start",
    │           extra=event_extra(
    │               event_type="fwd_bwd_start",  ──▶ record.log_type_name
    │               step=5,                     ──▶ record.step
    │               task_name=None,             ──▶ record.task_name
    │           ),
    │       )
    │
    │   # set_step() and add_step_tag() update trace-local state.
    ▼
_structured_logger  (logging.Logger, name="pimm.structured_logger",
                     propagate=False)
    │
    ▼
TraceEventsOnlyFilter  (drops records without a log_type_name)
    │
    ├── TraceJsonlHandler  ──▶ TraceJsonlFormatter ──▶
    │                         {output_dir}/structured_logs/*.jsonl
    └── Custom handler*    ──▶ Custom formatter    ──▶ external backend
```

## Custom handlers

`PIMM_STRUCT_LOGGER_HANDLERS` is a comma-separated list of fully qualified
Python function paths. When it is set, only those factories run.

```bash
export PIMM_STRUCT_LOGGER_HANDLERS="pimm.observability.structured_logger.jsonl_handler.register_jsonl_handler,mypackage.my_backend.register_my_db_handler"
```

A handler factory accepts the metadata forwarded by
`sl.init_structured_logger()` and attaches one handler to `structured_logger`.
For example:

```python
import logging
import os

from pimm.observability.structured_logger.jsonl_handler import TraceJsonlFormatter
from pimm.observability.structured_logger.structured_logging import (
    TraceEventsOnlyFilter,
)


class MyDBFormatter(TraceJsonlFormatter):
    """Enrich each record with backend-specific fields before serialization."""

    def _log_dict(self, record):
        data = super()._log_dict(record)
        data["cluster"] = os.environ.get("CLUSTER_NAME", "unknown")
        return data


class MyDBHandler(logging.Handler):
    """Send each trace event to a remote database as it is emitted."""

    def __init__(self, *, rank, source, world_size, attempt, job_id, trace_id):
        super().__init__()
        self.client = MyDBClient()
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

    def emit(self, record):
        try:
            self.client.insert_row(self.format(record))
        except Exception:
            self.handleError(record)


def register_my_db_handler(
    *,
    structured_logger,
    rank,
    source,
    world_size,
    attempt,
    job_id,
    trace_id,
    **kwargs,
):
    del kwargs
    structured_logger.addHandler(
        MyDBHandler(
            rank=rank,
            source=source,
            world_size=world_size,
            attempt=attempt,
            job_id=job_id,
            trace_id=trace_id,
        )
    )
```

## Analysis

Summarize a run:

```bash
pimm trace summarize <run-directory>
```

Export a Chrome Trace document:

```bash
pimm trace export <run-directory>
```

The default output is `<run-directory>/analysis/structured_trace.json`. Open it
in [Perfetto](https://ui.perfetto.dev/) or `chrome://tracing`.

The lower-level Python API is also available:

```python
from pimm.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
)


generate_gantt_trace(
    "outputs/structured_logs/",
    "outputs/analysis/structured_trace.json",
)
```
