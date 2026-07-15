# Distributed training

**Outcome:** run the same training config on more GPUs while preserving the
intended global batch and understanding what changes on resume.

## Supported strategies

| Strategy | Status | Behavior |
|---|---|---|
| single process | supported | no model wrapper when world size is one |
| DDP | supported/default | one process per GPU with PyTorch `DistributedDataParallel` |
| FSDP2 | experimental | composable `fully_shard`; CUDA required; not part of the same verified coverage as DDP |
| tensor/pipeline parallelism | not implemented | not a pimm execution strategy |

Do not use the FSDP2 path for a production-scale study without validating the
specific model, optimizer state, checkpoint save/load, and metric parity first.

## One process per GPU

`scripts/train.sh` uses `torchrun`. On two nodes with four GPUs each, Slurm
starts one launcher task per node and `torchrun` creates four local ranks per
node, for a world size of eight.

```bash
# one GPU on the current node
uv run pimm launch --train.config <config> --resources.nproc-per-node 1

# four GPUs on the current node
uv run pimm launch --train.config <config> --resources.nproc-per-node 4
```

Use `--dry-run` to inspect the exact `torchrun` command.

## Global batch and workers

`batch_size`, explicit validation/test batch sizes, and `num_worker` are global
totals. For global batch size $B$, global worker count $n_{\mathrm{workers}}$,
and world size $n_{\mathrm{ranks}}$, pimm assigns
$B/n_{\mathrm{ranks}}$ events and
$n_{\mathrm{workers}}/n_{\mathrm{ranks}}$ workers to each rank.

Explicit batch sizes must divide the world size. If `batch_size_val` or
`batch_size_test` is `None`, pimm uses one event per rank for that loader.

| Global config | 1 GPU | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|
| `batch_size=32` | 32/rank | 8/rank | 4/rank |
| `num_worker=16` | 16/rank | 4/rank | 2/rank |

Changing GPU count while holding the global batch fixed preserves the number of
optimizer steps per epoch, but it changes rank-local sampling and RNG streams.
It does not by itself guarantee bitwise-identical results.

## DDP config

DDP is the default when world size exceeds one:

```python
parallel = dict(strategy="ddp")
find_unused_parameters = False
sync_bn = False
```

Enable `find_unused_parameters` only for a model whose forward graph actually
leaves trainable parameters unused. `sync_bn=True` converts BatchNorm when
distributed on CUDA; most point models use other normalization layers.

## Experimental FSDP2

```python
parallel = dict(
    strategy="fsdp2",
    wrap_classes=["Block"],
)
```

The named classes are sharded before the root. This path uses a one-dimensional
FSDP device mesh and selects the structured checkpoint format. Treat the config
as an implementation hook, not a recommended recipe, until a model-specific
multi-rank integration test and parity result are published.

## Multi-node through Slurm

```bash
uv run pimm submit --site <site> \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config <config> \
  --dry-run
```

The generated Slurm request intentionally uses one task per node. Do not also
request one Slurm task per GPU; `torchrun` owns the GPU processes. GPU request
syntax (`gres` versus `gpus-per-node`) is site-specific.

## Resume after changing topology

The standard checkpoint can reshard model and optimizer state. Exact
mid-epoch loader continuation has a narrower condition:

| Change | Model/optimizer restore | Dataloader cursor | Consequence |
|---|---|---|---|
| same world size and workers | yes | restored when state is present | can continue mid-epoch |
| world size changes | resharded | skipped | saved epoch restarts; some batches may replay |
| workers per rank change | yes | skipped | saved epoch restarts; some batches may replay |
| `resume_strict_state=False` | best-effort | skipped | intended for recovery, not exact reproduction |

See {doc}`Checkpoint semantics <../operations/checkpoints>` for the complete
state contract.

## Validate scaling

Before a long job:

1. run the same tiny subset on one and two GPUs;
2. keep the global batch and seed fixed;
3. confirm finite loss, expected parameter load, metric aggregation, and one
   complete checkpoint;
4. interrupt and resume once on the same topology;
5. if topology changes are required, verify the epoch replay warning and account
   for it in the provenance record.

## Next

- {doc}`Slurm site profiles and submission <slurm>`.
- {doc}`Logging and resource diagnostics <../operations/logging>`.
- {doc}`Checkpoint and resume semantics <../operations/checkpoints>`.
