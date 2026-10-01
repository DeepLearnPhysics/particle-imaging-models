# Scale up

Run the same recipe on more GPUs, on Slurm, and across interrupted allocations.

## More GPUs on one machine

```bash
uv run pimm launch --train.config <recipe> --resources.nproc-per-node 4
```

`torchrun` starts one process per GPU, and pimm wraps the model in DistributedDataParallel when there is more than one. Locally, `--resources.nproc-per-node auto` uses every visible GPU. Batch sizes and worker counts are global and divide across GPUs; see [How pimm works](../get-started/how-pimm-works.md#batch-sizes-are-global).

| Global config | 1 GPU | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|
| `batch_size=32` | 32 per GPU | 8 per GPU | 4 per GPU |
| `num_worker=16` | 16 per GPU | 4 per GPU | 2 per GPU |

Experiment fields for distributed runs:

```python
parallel = dict(strategy="ddp")      # default
find_unused_parameters = False       # True only if some parameters get no gradient
sync_bn = False                      # convert BatchNorm layers when distributed
```

An experimental FSDP2 path shards the named module classes; it needs CUDA:

```python
parallel = dict(strategy="fsdp2", wrap_classes=["Block"])
```

## Run on Slurm

`pimm submit` takes the same flags as `pimm launch` and submits through Submitit. Site profiles in `launch/sites/` hold everything machine-specific:

| Profile | Machine |
|---|---|
| `local` | the current node, no container |
| `slurm` | generic Slurm base for your own profiles |
| `s3df`, `s3df-container` | SLAC S3DF, bare metal or Singularity |
| `nersc`, `nersc-container` | NERSC Perlmutter, bare metal or Shifter |

:::{warning}
`pimm submit` defaults to `--site s3df`, which sets SLAC's S3DF account (`mli:nu-ml-dev`), partition (`ampere`) and 512G of memory. Elsewhere, pass `--site slurm` or your own profile.
:::

A profile for your cluster:

```yaml
# launch/sites/mycluster.yaml
_base_: slurm.yaml
site: mycluster

paths:
  repo_root: /shared/me/particle-imaging-models
  exp_root: "{repo_root}/exp"

resources:
  scheduler: slurm
  nproc_per_node: 4
  cpus_per_proc: 12
  time: "12:00:00"
  mem: 192G
  account: <account>
  partition: <gpu-partition>
  gpu_directive: gres          # or gpus-per-node

container:
  runtime: none

env:
  PILARNET_DATA_ROOT_V3: /shared/data/pilarnet/v3
  HDF5_USE_FILE_LOCKING: "FALSE"
```

The checkout and experiment root must be visible from the compute nodes. Dry-run first; it prints the submission script, account, partition, GPU request and paths:

```bash
uv run pimm submit --site mycluster --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 --resources.time 00:10:00 --dry-run
```

Then submit without `--dry-run`. The command returns once the job is queued. Slurm output goes to `slurm_logs/slurm-<job id>.out` and the training log to the run directory.

For several nodes, set `--resources.nnodes`. pimm requests one Slurm task per node and `torchrun` starts the GPU processes on each node.

## Interactive allocations and chains

`--interactive` runs through `salloc` and `srun` and blocks the terminal:

```bash
uv run pimm submit --site mycluster --interactive --resources.qos <interactive-qos> \
  --resources.nproc-per-node 1 --resources.time 00:30:00 --train.config tests/tiny_semseg
```

With `--chain.jobs N` greater than one, pimm installs an `scron` watchdog instead. It requests N allocations one after another; each resumes from the newest complete checkpoint, and the chain stops when an attempt makes no checkpoint progress. The watchdog survives a dropped SSH session. It needs `scrontab` and a login-node QOS, such as NERSC's `cron`.

```bash
uv run pimm watchdog ls
uv run pimm watchdog rm <run-name>
```

The watchdog's log is `slurm_logs/watchdog/<run-name>/driver.log`.

## Launch recipes

Execution choices you repeat can live in `launch/runs/<name>.yaml`:

```yaml
run:
  name: sonata-v1
  timestamp: false
resources:
  nnodes: 2
  nproc_per_node: 4
  time: "12:00:00"
chain:
  jobs: 4
```

```bash
uv run pimm submit --site mycluster --recipe launch/runs/sonata-v1.yaml --train.config my_study/sonata_v1
```

## exex

exex is an execution package, pinned as a pimm dependency, that runs a job from a frozen copy of your source and carries its checkpoint from one attempt to the next. It is experimental. Turn it on for any `pimm launch` or `pimm submit` command:

```bash
PIMM_USE_EXEX=1 uv run pimm submit --site s3df --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 --dry-run
```

- **Source:** exex captures the Git-tracked files in `pimm/` and `configs/`, including uncommitted changes, and stages them on the target host. Dependencies must already be installed there.
- **Attempts:** `chain.jobs` counts the first attempt. Each attempt writes its own output directory; the completed checkpoint of one attempt is the input of the next. Crashes and failed checkpoints aren't retried.
- **W&B:** `PIMM_WANDB_HISTORY` sets how a resumed attempt treats tracking history: `new` (the default), `append` or `fork`.
- **Inspecting:** the launcher prints an experiment ID; from the checkout, run `exex experiments`, `exex status <id>` or `exex logs <id> 1`. exex keeps its catalog in `.exex/` and stages under `submit.folder`, which defaults to `<exp_root>/.exex`.
- **Remote submission:** `--submit.host nersc` with an absolute `--paths.exp-root` submits from S3DF to NERSC over your SSH configuration.

exex runs local jobs and Slurm jobs on one or more nodes. It doesn't run cloud jobs, array jobs or watchdogs.
