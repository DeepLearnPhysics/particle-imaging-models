# Slurm Submitit Requeue Launches

Use `pimm submit --chain.jobs N` for Slurm runs that need a stable experiment name
across short walltime allocations. pimm submits one submitit job; submitit
requeues it on timeout up to `N - 1` times.

Example:

```bash
pimm submit \
  --site nersc \
  --chain.jobs 4 \
  --run.name e050-tail-chain \
  --resources.nnodes 32 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Behavior:

- `--chain.jobs N` means at most `N` submitit attempts for the same stable run name.
- Attempt 1 starts normally unless `--train.resume` or `chain.resume_first=true` is set.
- Requeued attempts resume from the newest complete checkpoint.
- Each attempt gets a separate W&B run name by default, grouped under the stable
  run name.
- DCP checkpointing is the default for submitit-requeued runs.

To force the legacy checkpoint backend:

```bash
hooks.CheckpointSaverIteration.backend=torch
```
