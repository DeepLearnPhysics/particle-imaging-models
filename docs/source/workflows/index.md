# Workflows

Task-oriented guides for changing a verified run into a real experiment.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pimm-card-grid

:::{grid-item-card} Train or pretrain
:link: train
:link-type: doc
Choose a recipe, shrink it for a smoke test, then run the full configuration.
:::

:::{grid-item-card} Fine-tune
:link: fine_tune
:link-type: doc
Select compatible weights and verify which parameters actually loaded.
:::

:::{grid-item-card} Evaluate
:link: evaluate
:link-type: doc
Understand evaluator order, selection metrics, final testing, and provenance.
:::

:::{grid-item-card} Distribute
:link: distributed
:link-type: doc
Move from one to many GPUs while preserving global-batch semantics.
:::

:::{grid-item-card} Submit to Slurm
:link: slurm
:link-type: doc
Define a portable site profile, dry-run it, submit, monitor, and resume.
:::

::::

```{toctree}
:hidden:

train
fine_tune
evaluate
distributed
slurm
```
