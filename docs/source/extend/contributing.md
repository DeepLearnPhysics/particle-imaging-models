# Contributing

Contributions should leave a smaller, testable user contract: code, config,
documentation, provenance, and a bounded verification path.

## Development environment

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
uv sync --locked --group dev --group docs
```

The full dependency group targets Linux x86-64/Python 3.10 and the locked CUDA
stack. Pure launcher/docs changes can use the smaller environment where imports
permit it, but the final full docs build imports pimm registries.

## Repository map

```text
pimm/                 package: datasets, models, engines, launch/export utilities
configs/              versioned experiment recipes
launch/               execution defaults, sites, and run recipes
scripts/              train/test and dataset/site utilities
tests/unit/            fast isolated tests
tests/integration/     pinned-data/model and GPU/distributed workflows
docs/source/           MyST narrative and generated API entry pages
.github/workflows/     docs, GPU matrix, images, wheels, and release automation
```

## Run focused checks

```bash
# launcher/unit suite
uv run pytest -v tests/unit

# one test file or test
uv run pytest -v tests/unit/test_launch_rendering.py

# formatting check
uv run black --check pimm tests

# full docs (generates registry pages and imports pimm)
uv run make -C docs html
```

GPU/external-data tests are marked. The canonical tiny training, resume,
distributed, published-model, and metric-baseline paths live under
`tests/integration`; run only those for which the required hardware/data are
available, or use the repository's trusted GPU workflow.

## Documentation workflow

```bash
uv run make -C docs html
uv run make -C docs serve PORT=8000
```

Inspect the rendered page at desktop and narrow widths. Check the left
navigation, right TOC, tables, code overflow, dark mode, admonitions, and every
copyable command. A successful text build is not visual verification.

When a parser/registry/config fact can be generated, link to or generate it
instead of manually duplicating it. Runnable examples should use the pinned
mini fixture or another bounded, versioned asset.

## Scope a pull request

- Solve one reviewable problem; separate mechanical refactors from behavior.
- Preserve unrelated worktree changes.
- Add a regression test that fails before the fix.
- Update configs/docs when a public field, return schema, command, artifact, or
  scientific convention changes.
- Avoid checking in data, credentials, private paths, large outputs, or
  generated environment artifacts.
- Explain compatibility and migration for changed registered names or saved
  config/checkpoint behavior.

## Research additions

A new method or recipe should include:

- source paper/implementation and license;
- what is reproduced versus reimplemented or intentionally changed;
- dataset revision/split and preprocessing;
- a small correctness test and a bounded smoke recipe;
- evaluation definition and raw enough artifacts to audit it;
- known limitations, stability status, and hardware assumptions;
- citation and attribution in the relevant model/data page.

Do not label a result “reproduction” when preprocessing, data, metric, or
training budget is not matched. Mark unknown or pending measurements explicitly.

## Pull request checklist

- [ ] code/config and registry import are consistent;
- [ ] focused unit tests pass;
- [ ] relevant GPU/integration path is run or its absence is stated;
- [ ] `uv lock --check` passes and dependency changes are intentional;
- [ ] docs build and the rendered pages were inspected;
- [ ] copied commands and output/shape claims match the current code;
- [ ] scientific sources, licenses, revisions, and limitations are documented;
- [ ] no secret, private path, or unpublished data identifier is present.

Open an [issue](https://github.com/DeepLearnPhysics/particle-imaging-models/issues)
before a large cross-cutting addition so its interface and validation scope can
be agreed first.
