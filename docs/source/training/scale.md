# Scale up

Run the same recipe on more GPUs, on Slurm, and across interrupted allocations.

## More GPUs on one machine

```bash
uv run pimm launch --train.config <recipe> --resources.nproc-per-node 4
```

There is one process per GPU, started with `torchrun`, and with more than one GPU the model is wrapped in DistributedDataParallel. Locally, pass `--resources.nproc-per-node auto` to use every visible GPU. Batch sizes and worker counts are global and divided across GPUs; see [How pimm works](../get-started/how-pimm-works.md#batch-sizes-are-global).

| Global config | 1 GPU | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|
| `batch_size=32` | 32 per GPU | 8 per GPU | 4 per GPU |
| `num_worker=16` | 16 per GPU | 4 per GPU | 2 per GPU |

These are the experiment fields for distributed runs:

```python
parallel = dict(strategy="ddp")      # default
find_unused_parameters = False       # True only if some parameters get no gradient
sync_bn = False                      # convert BatchNorm layers when distributed
```

To shard the named module classes, use the experimental FSDP2 path, which requires CUDA:

```python
parallel = dict(strategy="fsdp2", wrap_classes=["Block"])
```

## Run on Slurm

`pimm submit` has the same flags as `pimm launch`, and jobs are submitted through Submitit. Site profiles in `launch/sites/` contain everything machine-specific:

| Profile | Machine |
|---|---|
| `local` | the current node, no container |
| `slurm` | default Slurm profile and base for your own profiles |
| `s3df`, `s3df-container` | example SLAC S3DF profiles, bare metal or Singularity |
| `nersc`, `nersc-container` | example NERSC Perlmutter profiles, bare metal or Shifter |

`--site slurm` is the default for `pimm submit`. Supply your account and partition with
`--resources.account` and `--resources.partition`, or set them in your own profile.

Here is a profile for your cluster:

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
  PIMM_DATA_ROOT: /shared/data/pimm
  HDF5_USE_FILE_LOCKING: "FALSE"
```

The checkout and experiment root must be visible from the compute nodes. Dry-run first to print the submission script, account, partition, GPU request and paths:

```bash
uv run pimm submit --site mycluster --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 --resources.time 00:10:00 --dry-run
```

Then submit without `--dry-run`. You get your shell back once the job is queued. Slurm output is written to `slurm_logs/slurm-<job id>.out` and the training log to the run directory.

For several nodes, set `--resources.nnodes`. The job has one Slurm task per node, and the GPU processes on each node are started with `torchrun`.

## Interactive allocations and chains

Add `--interactive` to run the job in the foreground through `salloc` and `srun`:

```bash
uv run pimm submit --site mycluster --interactive --resources.qos <interactive-qos> \
  --resources.nproc-per-node 1 --resources.time 00:30:00 --train.config tests/tiny_semseg
```

Set `--chain.jobs N` greater than one to install an `scron` watchdog instead. You then get N allocations, one after another; in each, training resumes from the newest complete checkpoint, and the chain stops after an attempt with no checkpoint progress. The watchdog is unaffected by a dropped SSH session. It needs `scrontab` and a QOS for its driver; set `--watchdog.qos` and `--watchdog.time` to match your cluster.

Without `--watchdog.qos`, the watchdog's QOS is the cluster's default.

```bash
uv run pimm watchdog ls
uv run pimm watchdog rm <run-name>
```

The watchdog's log is `slurm_logs/watchdog/<run-name>/driver.log`.

## Launch recipes

Keep execution choices you repeat in `launch/runs/<name>.yaml`:

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

exex is an execution package, pinned as a pimm dependency. With exex, you run a job from a frozen copy of your source, and its checkpoint carries over from one attempt to the next. exex is experimental. Turn it on for any `pimm launch` or `pimm submit` command:

```bash
PIMM_USE_EXEX=1 uv run pimm submit --site slurm --train.config tests/tiny_semseg \
  --resources.account <account> --resources.partition <gpu-partition> \
  --resources.nproc-per-node 1 --dry-run
```

- **Source:** the Git-tracked files in `pimm/` and `configs/`, including uncommitted changes, are captured and staged on the target host. Dependencies must already be installed there.
- **Attempts:** `chain.jobs` includes the first attempt. Each attempt has its own output directory; the completed checkpoint of one attempt is the input of the next. Crashes and failed checkpoints aren't retried.
- **W&B:** set `PIMM_WANDB_HISTORY` to choose what happens to tracking history in a resumed attempt: `new` (the default), `append` or `fork`.
- **Inspecting:** you get an experiment ID at launch; from the checkout, run `exex experiments`, `exex status <id>` or `exex logs <id> 1`. The exex catalog is in `.exex/`, and the staging directory is `submit.folder`, which is `<exp_root>/.exex` by default.
- **Remote submission:** pass `--submit.host login.cluster.example` and an absolute `--paths.exp-root` to submit on that host over your SSH configuration.

With exex, you can run local jobs and Slurm jobs on one or more nodes, but not cloud jobs, array jobs or watchdogs.
