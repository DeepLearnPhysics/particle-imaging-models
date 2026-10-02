# Contributing

Set up a development environment, run the tests, and open a pull request.

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
uv sync --locked --group dev --group docs
```

## Repository layout

```text
pimm/              the package: datasets, models, engines, launcher, export
configs/           experiment recipes
launch/            launcher defaults, site profiles and run recipes
scripts/           train and test entry scripts, dataset utilities
tests/unit/        fast tests
tests/integration/ tests that need a GPU, downloaded data or released models
docs/              this documentation
```

## Checks

```bash
uv run pytest -v tests/unit
uv run pytest -v tests/unit/test_launch_rendering.py
uv run black --check pimm tests
```

Integration tests are marked by what they need, such as `external_data` for downloads. Run the ones your hardware and data allow.

## Pull requests

Keep each pull request to one change, with a test that fails without it. Open an issue before a large architectural change. When you change a command, config field, output key or file format, update the recipes and docs in the same pull request.
