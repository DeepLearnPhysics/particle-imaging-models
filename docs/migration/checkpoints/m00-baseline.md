# Milestone 00 baseline evidence

Status: public gate passed; private compatibility pending

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

Resolved environment:

```text
Python 3.10.20
PyTorch 2.10.0+cu126
CUDA runtime 12.6
1 x NVIDIA A100-SXM4-40GB
```

Public focused test:

```text
uv run --frozen pytest -q tests/unit/migration/test_baseline_contracts.py
4 passed in 23.79s
```

Public full unit suite:

```text
uv run --frozen pytest -q tests/unit
103 passed, 19 warnings in 71.93s
```

The warnings are 18 upstream `torch.jit.script` deprecation warnings and one
PyTorch Geometric notice that `torch-cluster` is no longer necessary.

After the evidence files were updated, the focused boundary validation passed
again: `4 passed in 6.42s`.

The tested implementation commit is
`f26d2c42247cbabd13239d8fc19a1b914f4fcab7`. The boundary commit containing
this evidence is identified as `SELF` in `docs/migration/status.yaml`; its exact
SHA is the Git commit containing this file.

## Scope guard

- No WarpConvNet dependency has been added to PIMM.
- No file under `pimm/` has changed.
- No production config has changed.
- No legacy source has been deleted.
- Fixtures contain synthetic values only.
