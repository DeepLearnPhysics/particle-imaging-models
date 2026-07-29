# WarpConvNet migration

This document records the public migration boundary. The authoritative
execution specification is
`CODEX_PIMM_WCN_MIGRATION_PLAN_SINGLE_BRANCH.md` in the migration coordination
workspace; this repository contains only the public records that are safe to
publish.

## Status

Milestone 00 freezes the pre-migration repository state and observable
contracts. It does not add WarpConvNet, alter a model, change a configuration,
or replace the current packed batch format.

The migration uses one append-only logical branch in each repository:
`migration/warpconvnet`. Numbered milestones are commit boundaries on that
branch, not separate branches or pull requests. Public and private `main` remain
unchanged until the complete migration passes its final acceptance gate.

## Verified baselines

The following refs were fetched and verified on 2026-07-28:

| Role | Repository | `main` or pinned baseline |
| --- | --- | --- |
| public | `DeepLearnPhysics/particle-imaging-models` | `9491b0bf4b89bbee52a6383225a19f9c6a628a3c` |
| private | `DeepLearnPhysics/pimm-private` | `62239122c7a3112640743c8900cdc4336c33e59c` |
| backend | `NVlabs/WarpConvNet` | `cb22e75d1b102796585bcded5f4b02a492fb7fdd` |

The public baseline is the merge base of public and private `main`. It is a
real ancestor of private `main`; the private baseline is 110 commits ahead and
zero commits behind.

Annotated immutable baseline tags are:

- `pre-wcn-public-2026-07-28`
- `pre-wcn-private-2026-07-28`
- `pimm-wcn-upstream-base-2026-07-28`

The DeepLearnPhysics WarpConvNet integration fork and its
`migration/warpconvnet` branch were created from the pinned NVlabs commit.

## Repository direction

Public history may be merged into private history. Private history must never
be merged, rebased, cherry-picked with ancestry, or otherwise introduced into
the public branch. Public implementations inspired by private work are authored
as new public commits rooted in public `main`.

After the first public-to-private milestone synchronization, both migration
branches are append-only:

- merge an advancing `main` into the corresponding migration branch;
- never rebase, reset, or force-push a synchronized migration branch;
- merge the public migration branch into the private migration branch at every
  milestone boundary;
- never merge the private migration branch back into public.

The three repositories use physically separate clones and Git object stores.
The private clone has a fetch-only public remote whose push URL is disabled.

## Architectural boundary

WarpConvNet will be PIMM's primary sparse spatial implementation substrate. It
is not PIMM's universal public batch type. Dataset and trainer APIs remain
backend-neutral, and an ordinary `torch.nn.Module` remains a supported PIMM
model.

The accepted stable public backbone set after migration is:

1. PTv3 with maintained `m2` and `m8` profiles;
2. LitePT with current `LitePT-v1m2` semantics;
3. a dimension-generic 2D/3D Sparse U-Net;
4. one canonical Volt implementation with explicit presets.

The Panda detector is consolidated around current detector-v5m2 behavior.
Sonata and PoLAr-MAE remain public objectives. Private research objectives stay
in the private extension package.

CUDA is required for the supported migrated runtime. CPU-only model execution
is not a migration requirement.

## Milestone protocol

Every numbered milestone must end with:

1. focused implementation and tests;
2. a public unit-suite result;
3. an update to `docs/migration/status.yaml`;
4. a boundary commit on public `migration/warpconvnet`;
5. a non-fast-forward synchronization merge into private
   `migration/warpconvnet`;
6. the milestone's private compatibility evidence;
7. a private boundary commit and non-force push.

No legacy implementation is deleted before its model, checkpoint, scientific,
and performance gates pass. Active checkpoint and configuration inventory
items must have a converter or documented legacy execution path before their
only implementation is removed.

## Milestone 00 deliverables

This milestone adds only:

- architecture and contract records;
- public checkpoint inventory;
- migration status and decision records;
- deterministic synthetic fixtures for the current packed-batch and plain
  module contracts;
- tests that detect accidental drift in those records and fixtures.

The typed `SparseField`/`RaggedTensor` batch contract, import-path registry
resolution, WarpConvNet dependency, and all model migrations belong to later
milestones.
