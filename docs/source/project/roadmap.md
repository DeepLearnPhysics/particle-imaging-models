# Status and known limitations

This page describes the current repository. It is not a promise of dates or
future work. Pin a release or commit for research and validate the exact model,
data, and execution path you use.

## Current capability

| Area | Status |
|---|---|
| sparse, variable-length 3D point-cloud inputs | primary supported modality |
| local single-GPU and multi-GPU DDP training | supported path |
| Submitit-backed Slurm batch and interactive launch | implemented; site profile must be validated locally |
| standard split checkpoints and same-topology loader-state resume | implemented |
| topology-changing model/optimizer restore | implemented with saved-epoch restart rather than exact loader-cursor continuation |
| portable safetensors/Hub exports | implemented; loading reconstructs the model, not a preprocessing pipeline or trainer state |
| FSDP2 | experimental; no repository-wide model parity matrix |
| standalone evaluation | available through `scripts/test.sh`; no `pimm evaluate` command |
| 2D wire-plane, optical waveform, and other detector modalities | not currently supported by the common workflow |

## Platform boundary

The full locked training environment targets Linux x86-64, Python 3.10,
PyTorch 2.10, CUDA 12.6, and an NVIDIA driver compatible with that runtime.
macOS can use the launcher-only environment; the native sparse/CUDA stack is
not installed there. Other architectures, Python/PyTorch/CUDA combinations,
and non-NVIDIA accelerators are not covered by the committed lock/wheels.

## Scientific boundaries

- A common {py:class}`Point <pimm.models.utils.structure.Point>` representation does not standardize detector coordinates,
  units, calibration, feature order, labels, or selection.
- A committed recipe is not automatically a supported benchmark. Several are
  active research variants without signed-off metrics or stability guarantees.
- Published exports reconstruct architecture and weights, not the transform
  pipeline, class interpretation, dataset license, or evaluation protocol.
- PILArNet-M v1 and v2 have a standard downloader. The HDF5 reader accepts v3,
  but pimm does not provide a standard public v3 download path.
- The Parquet reader does not support `test_mode`; use the HDF5 reader for the
  current voxelized/augmented test path.
- Two committed Panda/Sonata Parquet configs inherit a base file absent from
  this revision and cannot currently resolve; see the {doc}`config catalog
  <../reference/configuration>`.
- Exact resume depends on dataloader state plus the same world-size/worker
  topology and does not guarantee deterministic kernels outside pimm's control.

## User-facing gaps that should stay visible

:::{admonition} TODO
:class: pimm-todo
Replace the gaps below only with measured values or an agreed project policy.
Until then, they remain visible here rather than being implied as features.

| Gap | What is needed before claiming completion |
|---|---|
| canonical software citation | approved authors/ORCIDs, `CITATION.cff`, release archive and DOI |
| model chooser metrics | signed-off held-out values, exact split/protocol, uncertainty, config and checkpoint revision |
| data reference | authoritative sizes/counts/checksums, licenses/citations, and coordinate/unit figures per revision |
| evaluation artifacts | one versioned prediction/metric schema per task plus a first-class evaluation CLI |
| FSDP2 support level | model-by-model multi-rank train/save/resume/evaluation parity matrix |
| support/security policy | maintainers, expected response scope, supported versions, private reporting route |
| compatibility policy | documented stability guarantees and migration/deprecation process for registry names/config/checkpoint schemas |
:::

## How priorities are chosen

No formal public roadmap or release schedule is committed in this repository.
Propose work through a [GitHub issue](https://github.com/DeepLearnPhysics/particle-imaging-models/issues)
with:

1. the user/research outcome;
2. the current blocking behavior;
3. the data/model/API and compatibility contract;
4. a bounded test or measurable acceptance criterion;
5. required ownership, hardware, data access, and publication decisions.

For implementation expectations, see {doc}`Contributing
<../extend/contributing>`.
