# Reproducibility

Reproducibility is an artifact contract, not a single seed. A result should be
traceable from data and code to predictions and metrics without relying on an
author's shell history.

## Minimum experiment bundle

| Item | Where pimm records it | What still needs explicit recording |
|---|---|---|
| code | `code/` snapshot and Git fields in `run_metadata.json` | release/commit and uncommitted patch state |
| configuration | `config.py`, `resolved_config.json`, `model_config.json` | rationale for experimental choices |
| command/execution | `run_metadata.json`, site/Slurm logs | site profile revision, container digest, job IDs |
| checkpoint | `model/` | immutable checksum or Hub commit |
| training log | `train.log`, TensorBoard/W&B | durable archive/identity |
| data | config roots/revision fields | file manifest, checksums, split construction, selection cuts, license |
| evaluation | evaluator config/logs | prediction artifact, metric definition, aggregation, uncertainty |

Do not publish absolute storage paths as provenance. Publish stable dataset and
checkpoint identifiers plus manifests/checksums.

## Determinism boundary

```python
seed = 0
deterministic = True
```

This requests deterministic setup and records the seed, but exact repetition
can still depend on CUDA kernels, PyTorch/native operator versions,
multiprocessing, input order, filesystem behavior, distributed topology, and
third-party libraries. Report the setting and environment; do not claim
bitwise reproducibility without testing it.

Rank seeds include rank/worker allocation. Changing world size or workers
changes streams. A structured resume on a changed topology restarts the saved
epoch instead of restoring the old loader cursor.

## Pin external artifacts

Use the repository form in commands and record the resolved Hub revision and
checksum alongside the run:

```text
hf://DeepLearnPhysics/Panda-Base
```

Record dataset repository type, revision, and file checksums. For containers,
record the registry tag and resolved image digest. For a pimm release, record
both `vX.Y.Z` and the Git commit.

## Baseline protocol

Before comparing an experimental change:

1. preserve one unmodified committed config;
2. use the same data revision/split/selection and evaluation implementation;
3. keep global batch, precision, schedule, and compute budget comparable;
4. run enough seeds for the claim and report individual values plus summary;
5. retain raw predictions or sufficient statistics to audit the metric;
6. report failed or excluded runs and the exclusion rule.

## Resume disclosure

If a run was interrupted, record:

- checkpoint chosen and saved epoch/iteration;
- same or changed world size/workers;
- whether the loader cursor restored;
- any replay-from-epoch warning;
- code/config/data changes between attempts;
- scheduler/job attempt IDs.

Do not describe a topology-changing recovery as “exact resume.” See
{doc}`Checkpoints <checkpoints>`.

## Model publication checklist

A useful model card includes:

- intended uses and limitations;
- model family, task head, parameter count, input/output schema;
- training data revision/split and licenses;
- exact transforms and feature order;
- training/evaluation configs and pimm version;
- metric definitions and results with uncertainty;
- hardware, precision, approximate training budget;
- checkpoint checksum/commit and license;
- example inference with expected output;
- method/dataset citations and contact/support route.

Unknown values should be marked unknown or pending, never inferred from a
neighboring experiment.

## Documentation quality gates

The highest-value user journeys should run in CI:

1. locked install and CLI dry run;
2. small public PILArNet-M-mini download and validation;
3. tiny single-GPU train/evaluate/checkpoint;
4. same-topology crash/resume;
5. two-GPU distributed smoke test;
6. published model load and metric baseline;
7. strict documentation build, internal links, and checked snippets.

The repository already exercises several of these under `tests/integration`.
Documentation should point to those canonical commands instead of maintaining a
second untested version.
