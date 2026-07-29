# Milestone 00 baseline evidence

Status: evidence pending

## Source state

- Public `main`: `9491b0bf4b89bbee52a6383225a19f9c6a628a3c`
- Private `main`: `62239122c7a3112640743c8900cdc4336c33e59c`
- WCN baseline: `cb22e75d1b102796585bcded5f4b02a492fb7fdd`
- Public/private merge base:
  `9491b0bf4b89bbee52a6383225a19f9c6a628a3c`
- Private divergence from public: zero behind, 110 ahead

## Environment

The public test environment is created from the checked-in lock:

```text
uv sync --frozen --group dev
```

Exact runtime and test results are recorded in the Milestone 00 boundary
commit after the focused and full unit suites complete.

## Scope guard

- No WarpConvNet dependency has been added to PIMM.
- No file under `pimm/` has changed.
- No production config has changed.
- No legacy source has been deleted.
- Fixtures contain synthetic values only.
