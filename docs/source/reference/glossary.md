# Glossary

## Data and geometry

**Event**
: One variable-length detector example. pimm normally models an event as a set
  of points; it does not assume a universal physical extent or unit.

**Point / hit**
: One row of point-aligned fields. A point can represent a simulated or
  reconstructed deposit depending on the dataset; “point” does not by itself
  specify detector semantics.

**`coord`**
: Floating-point coordinates with shape `(N, 3)`. Their frame, axis order, and
  units come from the dataset and transforms, not from the key name.

**`grid_coord`**
: Integer voxel coordinates used by sparse and serialized models. Usually
  derived from `coord` by a configured grid/voxel transform.

**`feat`**
: Model input features with shape `(N, C)`. The column order is a checkpoint
  contract and must be taken from its preprocessing config.

**Packed batch**
: Several variable-length events concatenated along the point dimension rather
  than padded to a common length.

**`offset`**
: Cumulative event-end indices in a packed batch. For `B` events it has shape
  `(B,)`; `offset[-1]` equals the total point count.

**`batch`**
: Per-point event indices, often derived from `offset`. Do not confuse this
  tensor with the configured event `batch_size`.

**Revision**
: A dataset or checkpoint version. Record an immutable commit/checksum when a
  mutable name such as `main`, `v2`, or `last` can resolve differently later.

**Split**
: The rule and membership defining train, validation, or test events. Equal
  split names across dataset revisions do not guarantee equal membership.

**Motif / semantic class**
: A point-wise topology label such as shower or track in PILArNet-M.

**PID**
: Particle-identification class. In Panda detector configs it is associated
  with particle instances, not merely a point-wise semantic label.

**Instance**
: Points grouped as one particle or interaction. Instance IDs are generally
  meaningful only inside one event.

## Models and learning

**Backbone / encoder**
: The representation-producing portion of a model, before a task-specific
  prediction head.

**Head / decoder**
: Task-specific layers that turn representations into logits, masks,
  regression outputs, or reconstruction targets. Config filenames use `dec`
  for a frozen-encoder adaptation in the committed Panda recipes.

**Pretraining**
: Training an objective intended to learn reusable representations, such as
  masked reconstruction or teacher/student prediction.

**Fine-tuning**
: Adapting pretrained parameters to a downstream task. **FFT** means full
  fine-tuning in the committed recipes; it is not a file format.

**Linear probe**
: Training a small prediction head while the representation model is frozen.
  It is a diagnostic of representation utility, not the same result as full
  downstream fine-tuning.

**PEFT**
: Parameter-efficient fine-tuning: train a limited set of adapters or task
  parameters rather than every model parameter. The exact trainable set must
  be verified from the config/model.

**Warm start**
: Load model parameters into a new run while starting new optimizer, scheduler,
  RNG, and loader state. `--train.weight` without resume is a warm start.

**Registry / `type`**
: A mapping from a config string such as `PT-v3m2` to a Python class or
  function. Registry names are case-sensitive public config identifiers.

## Experiments and execution

**Experiment config**
: Python under `configs/` describing what is trained or evaluated.

**Launch config / site profile**
: YAML under `launch/` describing where and how an experiment runs. A site
  profile captures machine policy; an optional launch recipe overlays a named
  execution choice.

**Run / experiment directory**
: The directory containing resolved config, metadata, source snapshot, logs,
  and checkpoints for one execution lineage.

**Global batch size**
: Total events across all ranks for one optimizer step. pimm's `batch_size`,
  explicit validation/test batch sizes, and `num_worker` are global totals.

**World size / rank / local rank**
: Total participating processes; one process's global index; and its index on
  the current node. pimm normally runs one rank per GPU.

**DDP**
: PyTorch DistributedDataParallel, pimm's default multi-rank strategy.

**FSDP2**
: PyTorch composable fully sharded data parallelism. The pimm path is
  experimental and requires model/checkpoint parity validation before research
  use.

**Hook**
: Ordered lifecycle behavior attached to a trainer, such as logging,
  evaluation, checkpoint loading, or checkpoint saving. Hook order can change
  semantics.

**Evaluator / tester**
: An evaluator runs validation logic during training; a tester drives a
  standalone task evaluation. Metric definitions and aggregation remain
  task-specific.

## Checkpoints and portability

**Training checkpoint**
: State used to continue a run: model, optimizer, scheduler/scaler, RNG,
  trainer progress, and—when supported—dataloader position.

**Standard / structured checkpoint**
: pimm's split `model/last/` layout: portable `weights.pth`, distributed
  trainer state under `trainer.dcp/`, and a `.complete` atomic-save marker.

**DCP**
: PyTorch Distributed Checkpoint. pimm uses it for reshardable trainer state;
  the raw `trainer.dcp/` directory is not a model export.

**Legacy checkpoint**
: Older monolithic `.pth` training state. It remains loadable where documented
  but does not have the standard split layout.

**Exact resume**
: Continuation with the saved loader cursor and stochastic/trainer state. It
  requires the saved state plus the same world-size and worker topology; a
  topology-changing restore can recover weights/optimizer but restarts the
  saved epoch and may replay batches.

**Model export**
: `model.safetensors` (or `model.bin`) plus an optional construction
  `config.json` and model card. It supports loading/inference/fine-tuning, not
  trainer resume.

**`model_best.pth` / `last`**
: `model_best.pth` is selected by the configured validation metric and is
  weights-only in the standard path. `model/last/` is the newest completed
  resumable checkpoint; “best” and “last” answer different questions.
