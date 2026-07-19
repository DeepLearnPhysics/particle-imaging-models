# Run on Slurm

**Outcome:** encode cluster-specific execution in one site profile, inspect the
rendered submission, and launch a portable training recipe.

`pimm launch` runs on the current node. `pimm submit` uses Submitit for a Slurm
batch job, or `salloc`/`srun` with `--interactive`.

## Resolution order

Later layers win:

```text
launch/defaults.yaml
    → launch/sites/<site>.yaml
    → optional --recipe YAML
    → launcher flags
    → post-`--` training-config overrides
```

Site YAML describes *where*. Python config describes *what*. Keep dataset
mounts, Slurm accounts, containers, and NCCL variables out of model configs.

## Create a site profile

```yaml
# launch/sites/mycluster.yaml
_base_: slurm.yaml
site: mycluster

paths:
  repo_root: /shared/me/particle-imaging-models
  exp_root: "{repo_root}/exp"

resources:
  scheduler: slurm
  nnodes: 1
  nproc_per_node: 4
  cpus_per_proc: 12
  time: "12:00:00"
  mem: 192G
  account: <account>
  partition: <gpu-partition>
  gpu_directive: gres  # or gpus-per-node on clusters that require it

container:
  runtime: none

env:
  PILARNET_DATA_ROOT_V2: /shared/data/pilarnet/v2
  HDF5_USE_FILE_LOCKING: "FALSE"
```

The checkout and experiment root must be visible from compute nodes. If the
image contains only the environment, set `container.repo_mount` and bind the
checkout there; `/opt/pimm/src` is the default.

## Dry-run first

```bash
uv run pimm submit --site mycluster \
  --train.config tests/tiny_semseg \
  --resources.nnodes 1 \
  --resources.nproc-per-node 1 \
  --resources.time 00:10:00 \
  --dry-run
```

Check all of the following before submission:

- account, partition/QOS, constraint, walltime, memory, and GPU directive;
- one Slurm task per node and expected GPUs per node;
- checkout, experiment, data, image, and bind paths;
- interpreter inside the image;
- rendezvous values and injected environment;
- final training config and fixed/timestamped run path.

## Submit a batch job

```bash
uv run pimm submit --site mycluster \
  --train.config my_study/sonata_v1 \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 12:00:00 \
  --run.name sonata-v1-seed17
```

The command returns after queueing. Slurm output defaults to
`slurm_logs/slurm-%j.out`; the experiment log is written under the run directory.

## Interactive allocation

```bash
uv run pimm submit --site mycluster \
  --interactive \
  --resources.qos <interactive-qos> \
  --resources.nproc-per-node 1 \
  --resources.time 00:30:00 \
  --train.config tests/tiny_semseg
```

A single interactive slot blocks in the terminal and ends if the SSH session
disappears. With `--chain.jobs N` greater than one, pimm instead installs a
`scron` watchdog that requests sequential `salloc` slots and resumes each slot
from the newest complete checkpoint. This mode requires a cluster with
`scrontab` and a login-node QOS such as NERSC's `cron` QOS.

```bash
uv run pimm submit --site nersc \
  --interactive \
  --chain.jobs 4 \
  --resources.qos interactive \
  --resources.time 02:00:00 \
  --train.config my_study/sonata_v1

uv run pimm watchdog ls
uv run pimm watchdog rm <run-name>
```

The watchdog survives a dropped SSH session or login-node restart. It retries
only when the previous slot wrote a newer complete checkpoint; repeated exits
without checkpoint progress mark the chain failed instead of looping forever.

## Reusable execution recipes

Store repeated execution choices in `launch/runs/<name>.yaml`:

```yaml
run:
  name: sonata-v1-seed17
  timestamp: false

resources:
  nnodes: 2
  nproc_per_node: 4
  time: "12:00:00"

chain:
  jobs: 4
```

```bash
uv run pimm submit --site mycluster \
  --recipe launch/runs/sonata-v1.yaml \
  --train.config my_study/sonata_v1 \
  --dry-run
```

Keep accounts, partitions, absolute dataset paths, and container locations in
the site profile so the recipe can move to another cluster.

## Monitor and recover

```bash
squeue -u "$USER"
scontrol show job <job-id>
tail -f slurm_logs/slurm-<job-id>.out
tail -f exp/<group>/<run>/train.log
```

To continue a fixed run after interruption:

```bash
uv run pimm submit --site mycluster \
  --train.config <same-config> \
  --run.name <same-run> \
  --run.no-timestamp \
  --train.resume
```

For a chain, each attempt must target the same run and enable resume after the
first. Inspect the rendered attempts; scheduler requeue behavior and signal
delivery differ by site.

For a chained interactive run, inspect
`slurm_logs/watchdog/<run-name>/driver.log`. Removing the watchdog stops future
supervision; pass `--scancel` to also cancel the current driver and allocation.

## Cluster checklist

- Pin pimm and container releases for published work.
- Test one rank before requesting multiple nodes.
- Put the data root and experiment root on filesystems intended for that access
  pattern.
- Keep secrets in scheduler/site secret mechanisms, not committed YAML.
- Record job IDs, resolved config, site profile revision, and container digest.
- Read {doc}`Checkpoint semantics <../operations/checkpoints>` before changing
  GPU or worker topology on resume.

## Next

- {doc}`Distributed semantics <distributed>`.
- {doc}`Logging and diagnostics <../operations/logging>`.
- {doc}`Troubleshooting <../operations/troubleshooting>`.
