# Reference

Use these pages when you know the object you need and want its exact spelling,
location, or behavior. For an end-to-end path, start with a
{doc}`workflow <../workflows/index>` instead.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Command line
:link: cli
:link-type: doc
The three `pimm` commands, launcher flag groups, override boundary, and the
standalone evaluation script.
:::

:::{grid-item-card} Configurations
:link: configuration
:link-type: doc
Experiment-versus-launch precedence, filename vocabulary, and every committed
recipe grouped by task.
:::

:::{grid-item-card} Environment variables
:link: environment
:link-type: doc
Data roots, checkpoint/cache locations, credentials, distributed variables,
and which process reads each one.
:::

:::{grid-item-card} Glossary
:link: glossary
:link-type: doc
Short definitions for packed point clouds, model/training terms, launch terms,
and checkpoint semantics.
:::

:::{grid-item-card} Python API
:link: ../api/index
:link-type: doc
Generated registries plus the builders, point structure, configuration loader,
checkpoint manager, and export functions.
:::

::::

:::{note}
Parser help and generated API pages are the authority for the installed
version. If a table here disagrees with `pimm <command> --help`, report it as a
documentation bug and follow the parser output.
:::

```{toctree}
:hidden:
:maxdepth: 1

Command line <cli>
Configurations <configuration>
Environment variables <environment>
Glossary <glossary>
Python API <../api/index>
```
