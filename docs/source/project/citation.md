# Citation

A pimm result normally needs three kinds of citation:

1. the exact pimm software release/commit;
2. every model/method implementation used by the config;
3. the dataset and detector/simulation sources used to create the events.

Do not cite only the repository when the scientific claim depends on a method
or dataset paper.

## Cite the exact software

This repository does **not currently contain `CITATION.cff` or a canonical
software DOI**. Until the software-citation TODO below is completed, cite the
release URL and commit used by the experiment and retain the resolved config.

```bash
uv run python -c 'from importlib.metadata import version; print(version("pimm"))'
git rev-parse HEAD
git status --short
```

Use this explicitly non-canonical template only when your venue requires
BibTeX; replace every bracketed value:

```bibtex
@software{pimm_[release_or_commit],
  author  = {DeepLearnPhysics contributors},
  title   = {Particle Imaging Models (pimm)},
  version = {[release or full commit]},
  year    = {[release year]},
  url     = {https://github.com/DeepLearnPhysics/particle-imaging-models/tree/[tag-or-commit]},
  note    = {Git commit [full SHA]}
}
```

:::{admonition} TODO
:class: pimm-todo
Add `CITATION.cff`, named authors/ORCIDs, a preferred software title, release
archive/DOI, and a stable BibTeX entry. Replace the template above once those
values are approved.
:::

## Find method and dataset citations

| Config family | Source of truth to consult | Current documentation status |
|---|---|---|
| Panda / Sonata | linked model card, recipe comments, and the implementation lineage in `pimm/models/sonata/` | canonical paper/BibTeX is not recorded centrally in this repository |
| Panda Detector | model card plus detector/backbone sources | see TODO below |
| PoLAr-MAE | [PoLAr-MAE source repository](https://github.com/DeepLearnPhysics/PoLAr-MAE) and released model card | pimm configs preserve compatibility, but this repository has no central citation manifest |
| Point Transformer V3 backbone | [Point Transformer V3 paper](https://arxiv.org/abs/2312.10035) and upstream implementation | cite it when the selected config builds a PTv3 backbone |
| PILArNet-M | [canonical dataset card](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M) and its linked publication | license/citation should be copied into a future versioned data manifest |

:::{admonition} TODO
:class: pimm-todo
Publish a reviewed citation mapping for Panda Detector and other research
configs that do not yet have canonical method metadata in the repository.
:::

If a source does not provide citation metadata, mark it as pending in your
internal record and resolve it with the model/dataset owner before publication.
Do not infer a citation from a similarly named project.

## Record the configuration behind a result

Alongside the bibliography, preserve:

- pimm version, full commit, and uncommitted diff status;
- `resolved_config.json`, source snapshot, and exact command;
- checkpoint repository/file plus immutable revision and checksum;
- dataset repository, revision, split, selection, and file manifest;
- preprocessing, coordinate/feature units and order, target class order;
- metric definition, aggregation, evaluation config, and prediction artifact.

The {doc}`Reproducibility guide <../operations/reproducibility>` gives the full
artifact contract.

## Attribution and license

pimm is distributed under the
[MIT License](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/LICENSE).
Vendored or adapted components can carry their own license and citation
requirements; check their source directories and upstream projects. Scientific
citation is still required when code reuse is permitted by a software license.

## Before submitting a paper

- [ ] software release and commit are immutable and reported;
- [ ] method/backbone/task-head citations match the actual resolved model;
- [ ] dataset/simulation citation matches the exact revision and split;
- [ ] every pending placeholder on the relevant model/data page is resolved or
  explicitly disclosed;
- [ ] reported metrics can be traced to a saved evaluation config and artifact.
