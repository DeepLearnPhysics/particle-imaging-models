# Tutorials

Tutorials connect the data, model, and evaluation guides into complete research
tasks. Start by exploring a released checkpoint on PILArNet-M-mini, or complete
the {doc}`first experiment <../getting_started/quickstart>` before training.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pimm-card-grid

:::{grid-item-card} Explore Panda
:link: explore_panda
:link-type: doc
Inspect one real test event and run the released encoder, semantic, particle,
and interaction models with interactive Plotly figures.
:::

:::{grid-item-card} Explore PoLAr-MAE
:link: explore_polarmae
:link-type: doc
Inspect semantic predictions, masked reconstruction, and token embeddings.
:::

:::{grid-item-card} Train semantic segmentation
:link: semantic_segmentation
:link-type: doc
Adapt the verified tiny pipeline into a per-point classification experiment.
:::

:::{grid-item-card} Parameter-efficient fine-tuning
:link: peft
:link-type: doc
Freeze a base model, inject LoRA into selected projections, and verify the
trainable set.
:::

::::

```{toctree}
:hidden:

explore_panda
explore_polarmae
semantic_segmentation
peft
```
