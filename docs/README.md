# pimm documentation

Source for <https://deeplearnphysics.org/particle-imaging-models/stable/>.
The site uses Sphinx, MyST Markdown, `sphinx-book-theme`, and autodoc. A full
build imports pimm so the narrative pages and API reference stay synchronized
with the checked-out code.

## Build and serve

From the repository root:

```bash
uv sync --locked --group docs
uv run --no-sync make -C docs html
uv run --no-sync make -C docs serve  # http://localhost:8000
```

Use the local server rather than opening `index.html` directly. It gives
embedded Plotly figures and relative assets the same URL behavior as the
published site.

Useful checks:

```bash
uv run --no-sync make -C docs clean
uv run --no-sync make -C docs linkcheck
uv run --no-sync sphinx-build -W --keep-going -b html docs/source docs/build/html
```

`make html` runs `docs/gen_api.py` first. It imports the live registries and
regenerates the API pages under `source/api/`; use the full Linux project
environment or the project container for that build.

## Structure

```text
docs/
├── Makefile
├── gen_api.py
└── source/
    ├── index.md             landing page and global navigation
    ├── getting_started/     installation, first run, mental model
    ├── workflows/           train, fine-tune, evaluate, distributed, Slurm
    ├── models/              released checkpoints, inference, export
    ├── data/                conventions, PILArNet-M, transforms, custom data
    ├── operations/          configs, checkpoints, logging, reproducibility
    ├── tutorials/           runnable scientific walkthroughs
    ├── extend/              models, losses, datasets, transforms, hooks
    ├── reference/           CLI, config keys, environment, glossary
    ├── api/                 generated Python API and registries
    ├── project/             support, roadmap, citation
    ├── _static/             theme assets and checked-in tutorial figures
    └── _templates/          autodoc templates
```

Keep one topic in one canonical page and link to it elsewhere. Put commands
before detailed explanation, link named pimm classes and functions to the API,
and use inline MathJax for short equations. Unknown scientific values or
unavailable plots should be marked with a red `TODO` admonition rather than
filled with an estimate.

## Runnable tutorials and figures

`tutorials/explore_panda.py` and `tutorials/explore_polarmae.py` are
Jupytext-compatible `# %%` notebooks. They are the source of the interactive
HTML, static PNG fallbacks, and JSON metadata in `_static/tutorials/`.

Run them as ordinary Python:

```bash
# PoLAr-MAE: CPU or CUDA
uv run --group train --with plotly \
  python docs/source/tutorials/explore_polarmae.py --models all --device cpu

# Panda dataset views: CPU
uv run --group train --with plotly \
  python docs/source/tutorials/explore_panda.py --models dataset

# Panda released models: CUDA
uv run --group train --with plotly \
  python docs/source/tutorials/explore_panda.py --models all --device cuda
```

Or convert exactly the same sources to notebooks:

```bash
uv run --group train --group dev --with jupytext \
  jupytext --to ipynb docs/source/tutorials/explore_panda.py
```

The manual **Docs figures** workflow runs both sources on a Modal L4 and
uploads the generated directory as a review artifact. It never commits or
publishes figures automatically. After downloading the artifact:

1. compare its JSON metadata and plots with the intended event, transforms,
   checkpoint repositories, and seed;
2. visually inspect the interactive and static versions;
3. replace `_static/tutorials/` and commit the source and generated assets
   together;
4. run the strict Sphinx build.

The normal docs build embeds checked-in results and performs no data download
or model inference.
