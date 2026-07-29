# ADR 0003: inventory and deletion gates

- Status: accepted
- Date: 2026-07-28
- Milestone: 00

## Context

Public releases and private experiments depend on architecture versions that
will be consolidated. Deleting a source copy before identifying its active
configs and checkpoints can leave the only executable definition of a
scientific result unavailable.

## Decision

Maintain machine-readable public and private checkpoint/config inventories.
Every active checkpoint must end with either:

1. a passing converter plus migrated configuration; or
2. a documented immutable legacy-release execution path.

Unknown activity or disposition is recorded explicitly as `UNDECIDED`. It is
not silently treated as dead. Destructive migration is blocked while an active
item remains undecided or while its selected path has not passed acceptance.

Synthetic Milestone 00 fixtures are small text files containing no detector
data, credentials, checkpoints, or private paths. They freeze observable
contracts without changing production behavior.

## Consequences

- Inventory gaps are visible migration blockers.
- Archive candidates are not deleted merely because they look historical.
- Checkpoint acceptance includes key conversion, numerical/model parity, and
  the relevant scientific metrics.
- The fixture files can be reviewed and regenerated without external data.
