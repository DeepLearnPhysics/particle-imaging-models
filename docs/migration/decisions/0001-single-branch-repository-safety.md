# ADR 0001: single-branch repository safety

- Status: accepted
- Date: 2026-07-28
- Milestone: 00

## Context

PIMM's public repository is the canonical implementation line. The private
repository contains public history plus private research. Moving private Git
ancestry into the public repository would disclose private history and break
the strict-superset relationship.

The migration lasts for multiple milestones, while unrelated work may continue
on each repository's `main`.

## Decision

Use one long-lived branch named `migration/warpconvnet` in each of the public,
private, and DeepLearnPhysics WarpConvNet repositories.

The public branch starts at public `main`
`9491b0bf4b89bbee52a6383225a19f9c6a628a3c`. The private branch starts at
private `main` `62239122c7a3112640743c8900cdc4336c33e59c`. The WCN branch starts
at pinned upstream commit
`cb22e75d1b102796585bcded5f4b02a492fb7fdd`.

Public history is merged into private history at every milestone. Private
history is never merged into public. After the first synchronization, migration
branches are append-only: no rebase, reset, or force-push. Advancing `main`
branches are incorporated with explicit merge commits.

The repositories use physically separate clones. The private clone's public
remote has a disabled push URL.

## Consequences

- Milestone boundaries remain reviewable in one draft public pull request.
- Public commits have no private ancestors.
- Conflict resolution for private extensions stays private.
- Rollback and review use boundary commits rather than disposable branches.
- `main` remains free of migration commits until final acceptance.
