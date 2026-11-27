# Slurm Submitit Requeue Launches

Use `pimm submit --chain.jobs N` for Slurm runs that need a stable experiment name
across short walltime allocations. pimm submits one submitit job; submitit
requeues it on timeout up to `N - 1` times.

Example:

```bash
pimm submit \
  --site slurm \
  --resources.account <account> \
  --resources.partition <gpu-partition> \
  --chain.jobs 4 \
  --run.name pretrain-chain \
  --resources.nnodes 32 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Behavior:

- `--chain.jobs N` means at most `N` submitit attempts for the same stable run name.
- Attempt 1 starts normally unless `--train.resume` or `chain.resume_first=true` is set.
- Requeued attempts resume from the newest complete checkpoint.
- Each attempt gets a separate W&B run name by default, grouped under the stable
  run name.
- DCP checkpointing is the default.

To select the legacy checkpoint format, add this training override after `--`:

```bash
checkpoint_format=legacy
```
