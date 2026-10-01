# Glossary

event
: One detector readout, stored as a variable number of points.

point
: One row of an event: a position plus its features and labels.

`coord`
: Point positions, shape (N, 3) for 3D readers. Their frame and units come from the dataset and the transforms.

`grid_coord`
: Integer grid cells derived from `coord` by grid sampling. Sparse-convolution and serialized-attention models read them.

`feat`
: Model input features, shape (N, C), in the order of `Collect`'s `feat_keys`.

packed batch
: Several events concatenated along the point dimension instead of padded.

`offset`
: The cumulative end index of each event in a packed batch; `offset[-1]` is the total number of points.

revision
: A version of a dataset, such as PILArNet-M `v3`, or of a Hugging Face repository.

semantic class
: A per-point label such as shower or track (`segment_motif` in PILArNet-M).

PID
: Particle identity, such as muon or proton (`segment_pid` in PILArNet-M).

instance
: The points of one particle or one interaction. Instance ids are unique within an event.

backbone
: The encoder that turns a point cloud into per-point features.

head
: The task-specific layers on top of a backbone.

pretraining
: Training on unlabeled data, such as Panda's self-distillation or PoLAr-MAE's masked point modeling.

fine-tuning
: Training a pretrained model on a labeled task. In recipe names, `fft` means full fine-tuning, `dec` the decoder and head, and `lin` the head only.

warm start
: Loading weights into a new run, with a new optimizer and step count (`--train.weight`).

resume
: Continuing a run from its checkpoint, including optimizer, step and data-loader state (`--train.resume`).

registry
: A mapping from a name in a config's `type` field to a Python class.

experiment config
: Python under `configs/` that says what to train.

site profile
: YAML under `launch/sites/` that says where and how to run.

global batch size
: Events per optimizer step, summed over all GPUs. pimm's `batch_size` is global.

rank
: One training process, normally one per GPU.

DDP
: PyTorch DistributedDataParallel, pimm's default for more than one GPU.

export
: A portable model directory with weights and a config, made by `pimm export`.
