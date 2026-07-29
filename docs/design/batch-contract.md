# Batch contract

Milestone 00 freezes the current packed point-batch behavior and records the
accepted backend-neutral destination. It does not introduce the destination
types.

## Baseline packed mapping

At public commit `9491b0bf4b89bbee52a6383225a19f9c6a628a3c`,
`pimm.datasets.utils.collate_fn` recursively collates flat dataset mappings:

- tensor leaves are concatenated on dimension zero;
- strings become lists;
- mapping keys beginning with `_` are omitted;
- keys containing `offset` are converted from per-sample cumulative offsets to
  cumulative offsets for the full batch;
- sequence samples receive a cumulative length tensor;
- unsupported leaves are delegated to PyTorch's `default_collate`.

A common trainer-facing batch is:

```python
{
    "coord": Tensor[N, D],
    "feat": Tensor[N, C],
    "offset": Tensor[B],
    "segment": Tensor[N],
    "name": list[str],
}
```

`offset[i]` is the exclusive end of event `i`. The deterministic
`legacy_packed_batch.yaml` fixture freezes concatenation, offset construction,
metadata handling, and private-key omission.

The baseline collator concatenates arbitrary tensors. It does not distinguish
fixed-shape tensors from ragged point-aligned tensors by type. That behavior is
retained through a later `LegacyPackedCollator`; it is not the destination
default.

## Accepted backend-neutral destination

The migrated public batch remains a plain nested mapping:

```python
batch = {
    "inputs": {
        "points": SparseField(...),
    },
    "targets": {
        "segment": RaggedTensor(...),
    },
    "meta": {
        "batch_size": int,
        "name": list[str],
    },
}
```

`SparseField` owns aligned coordinates, features, CPU `int64` row splits, and
optional aligned input attributes. `RaggedTensor` owns arbitrary variable-length
values and CPU `int64` row splits. Scientific supervision belongs under
`targets`, not in a WarpConvNet geometry object.

The mapping may contain arbitrary additional modalities. In particular,
independent 2D wire planes are represented as independent sparse fields:

```python
batch["inputs"]["planes"] = {
    "u": SparseField(...),
    "v": SparseField(...),
    "y": SparseField(...),
}
```

Plane identity is not encoded as a fake third spatial coordinate.

## Backend boundary

Datasets do not return WarpConvNet `Points` or `Voxels` as PIMM's universal
batch ontology. Conversion to backend geometry happens at an explicit model or
backend seam after the trainer moves the batch to its device.

Milestone 02 introduces the typed structures and type-directed collator.
Milestone 04 introduces the WarpConvNet conversion seam. Neither is implemented
by Milestone 00.

## Compatibility and deletion gate

Existing configs continue to use the current packed dictionary and collator
until individually migrated. Legacy collation cannot be deleted until active
public and private configs have a recorded typed-batch migration or a documented
legacy execution path.
