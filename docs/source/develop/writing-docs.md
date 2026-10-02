# Writing docs

The docs are MyST Markdown built with Sphinx. Reference pages are generated from the code; the rest is written by hand, under the rules below.

## Build

```bash
uv sync --locked --group docs
uv run --no-sync make -C docs html
uv run --no-sync make -C docs serve
```

`make html` runs `docs/tools/gen_reference.py`, then Sphinx with `-W`, so any warning fails the build. `make serve` serves the result at http://localhost:8000.

The generator writes [Command line](../reference/cli.md) from the `pimm` parsers, [Recipe catalog](../reference/recipes.md) by loading every config, [Python API](../reference/api.md) from the registries, and [Environment variables](../reference/environment.md) from the variables the code reads. To change those pages, change the code: help text, docstrings and config docstrings.

## Rules

1. **One question per page, answered first.** Open with a working example and its output; explanation follows.
2. **One home per concept.** Global batch sizes, the event format and the checkpoint layout each live on one page. Other pages link to it.
3. **Generate what the code knows.** Commands, flags, recipes, registered names and environment variables come from the code.
4. **Every example runs.** Commands use `uv run` and the mini dataset, and show what you'll see.
5. **Broken means the build fails.** Build with `-W` on every pull request. Unfinished work goes in an issue, not on a page.
6. **Say what's there.** Describe what each model, dataset, recipe and command is and does. Results belong in papers, not in the docs.
7. **Write to the reader.** Second person, present tense, active verbs. Define a term where it first appears.
8. **Navigation fits in one row.** At most six top-level sections, labels that match URLs, previous and next links on every page.
9. **Site-specific material stays labeled.** Examples run anywhere; S3DF and NERSC details live in site profiles and in the sections about them.
10. **Length is a cost.** A how-to page runs to about 800 words. Tables compare things; they don't replace sentences.
