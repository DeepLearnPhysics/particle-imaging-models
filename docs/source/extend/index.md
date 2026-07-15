# Extend pimm

Start with the system map, then add the smallest component that expresses the
new behavior.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pimm-card-grid

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc
Registries, object ownership, batch/model contracts, hook lifecycle, and public
extension seams.
:::

:::{grid-item-card} Contributor setup
:link: contributing
:link-type: doc
Environment, tests, documentation build, PR scope, and research provenance.
:::

:::{grid-item-card} Add a model
:link: add_model
:link-type: doc
Choose config versus backbone versus top-level model, then test forward/loss/output contracts.
:::

:::{grid-item-card} Add a dataset
:link: add_dataset
:link-type: doc
Register a lazy reader, validate packed collation, and document scientific fields.
:::

:::{grid-item-card} Add a transform
:link: add_transform
:link-type: doc
Preserve point and auxiliary-target alignment through deterministic tests.
:::

:::{grid-item-card} Add a hook
:link: add_hook
:link-type: doc
Use the narrowest lifecycle event and make distributed/state behavior explicit.
:::

::::

```{toctree}
:hidden:

architecture
contributing
add_model
add_dataset
add_transform
add_hook
```
