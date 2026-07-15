# Run and reproduce

Operational guides for changing an experiment without losing its provenance or
misstating what an interrupted run did.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pimm-card-grid

:::{grid-item-card} Configuration
:link: configuration
:link-type: doc
Inheritance, precedence, overrides, common fields, and resolved artifacts.
:::

:::{grid-item-card} Checkpoints and resume
:link: checkpoints
:link-type: doc
On-disk formats, atomic saves, warm starts, exact-resume boundary, and topology changes.
:::

:::{grid-item-card} Logging and diagnostics
:link: logging
:link-type: doc
Console, TensorBoard, W&B, health monitors, and profiling.
:::

:::{grid-item-card} Reproducibility
:link: reproducibility
:link-type: doc
The minimum artifact and metadata bundle behind a defensible result.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc
Find an exact symptom, run one diagnostic, and apply the narrow fix.
:::

::::

```{toctree}
:hidden:

configuration
checkpoints
logging
reproducibility
troubleshooting
```
