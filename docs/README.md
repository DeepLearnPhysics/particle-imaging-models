# pimm documentation

Source for <https://deeplearnphysics.org/particle-imaging-models/stable/>: MyST Markdown built with
Sphinx and the PyData Sphinx Theme.

## Build

From the repository root:

```bash
uv sync --locked --group docs
uv run --no-sync make -C docs html    # generate the Reference pages, then build with -W
uv run --no-sync make -C docs serve   # http://localhost:8000
```

## Layout

```text
docs/
├── Makefile
├── tools/gen_reference.py   writes the generated Reference pages
└── source/
    ├── conf.py
    ├── index.md
    ├── get-started/  models/  data/  training/  develop/  reference/
    └── _static/             CSS, logos, pimm.js (landing logo, 3D event viewer), figure data
```

`tools/gen_reference.py` writes `reference/cli.md`, `recipes.md`, `api.md` and `environment.md` from
the `pimm` parsers, the configs, the registries and the environment variables the code reads. They
are not committed: to change them, change the code.

The writing rules are in `source/develop/writing-docs.md`.
