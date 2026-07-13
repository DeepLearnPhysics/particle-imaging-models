# pimm Launch Configs

`pimm launch` runs training locally or inside an existing allocation.
`pimm submit` submits a managed Slurm job through submitit.

Python training configs remain the source of truth for model and training
behavior. Launch YAML describes execution policy: site resources,
container/runtime, checkpoint/resume choices, run naming, and environment.

## Local Or Allocated Runs

Run on one local process:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Run on four local GPUs:

```bash
pimm launch \
  --resources.nproc-per-node 4 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Run from inside a user-authored Slurm allocation:

```bash
srun pimm launch \
  --resources.nnodes "$SLURM_NNODES" \
  --resources.nproc-per-node "$SLURM_GPUS_ON_NODE" \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Training config overrides go after `--` as plain `key=value` arguments:

```bash
pimm launch \
  --resources.nproc-per-node 4 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=1000 batch_size=8
```

Use `--dry-run` to print the rendered local launch script.

## Managed Slurm Submission

Submit through the site-aware submitit path:

```bash
pimm submit \
  --site s3df \
  --resources.nnodes 1 \
  --resources.nproc-per-node 4 \
  --resources.time 00:30:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=1000 batch_size=8
```

For NERSC:

```bash
pimm submit \
  --site nersc \
  --resources.nnodes 1 \
  --resources.nproc-per-node 4 \
  --resources.time 00:30:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Use `--dry-run` to print the submitit manifest and `--output PATH` to write it.
`--submit.host iana` can be used when submission should happen from a remote login host.

## Container Repo Mounts

The Docker images ship only the locked environment - no pimm source is baked
in. Containerized launch configs bind the host checkout at `paths.repo_root`
onto `container.repo_mount`, which defaults to `/opt/pimm/src`, and run
`scripts/train.sh` from that mounted path, so `pimm launch`, imports, and
training code always come from the user's clone.

Manual Apptainer/Singularity use should preserve the same mount:

```bash
apptainer exec --nv \
  --bind "$PWD:/opt/pimm/src" \
  --pwd /opt/pimm/src \
  /path/to/pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Manual Shifter use should do the equivalent volume mount:

```bash
shifter --image=youngsm/pimm-nersc:pytorch2.10.0-cuda12.6 \
  --volume="$PWD:/opt/pimm/src" \
  /bin/bash -lc 'cd /opt/pimm/src && pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask'
```

## Requeue Attempts

Use `--chain.jobs N` for managed short-walltime runs. pimm submits one submitit job,
and submitit requeues it on timeout up to `N - 1` times.

```bash
pimm submit \
  --site nersc \
  --chain.jobs 4 \
  --resources.nnodes 32 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Attempt 1 starts normally unless resume is requested. Requeued attempts resume
from the newest complete checkpoint in the stable experiment directory.

## Checkpoint Backend Policy

`pimm launch` and `pimm submit` default `CheckpointSaverIteration` to DCP when
the rendered run is multi-rank, submitit-requeued, or requests
`parallel.strategy=fsdp2`.

Plain `.pth` checkpoints remain supported for local/simple runs, legacy loading,
and export-style artifacts. To force the legacy backend:

```bash
pimm submit \
  --site s3df \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- hooks.CheckpointSaverIteration.backend=torch
```

## File Ownership

- `launch/defaults.yaml`: common launcher defaults.
- `launch/sites/slurm.yaml`: generic Slurm defaults for resources, logs, and
  environment; cluster-specific Slurm sites inherit from this with `_base_`.
- `launch/sites/s3df.yaml`: S3DF account/partition and environment variables;
  bare metal, jobs run in the checkout's uv-managed `.venv`.
- `launch/sites/s3df-container.yaml`: containerized S3DF alternative
  (Singularity, frozen image environment) for large-scale or pinned runs.
- `launch/sites/nersc.yaml`: NERSC account/qos/constraint and Perlmutter
  environment variables; bare metal, using the `.venv` prepared once with
  `scripts/nersc_env.sh`.
- `launch/sites/nersc-container.yaml`: containerized NERSC alternative
  (Shifter, frozen image environment) for large-scale or pinned runs.
- `container.repo_mount`: in-container path where `paths.repo_root` is mounted
  so `pimm` imports resolve to the checkout; defaults to `/opt/pimm/src`.
- `launch/sites/local.yaml`: no scheduler/container wrapper; runs directly on
  the current node.
- `launch/runs/*.yaml`: optional named launch recipes focused on execution
  choices, not model architecture.
