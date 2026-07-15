# Support

Start from the route that matches the problem. pimm currently provides public
GitHub issue tracking; the repository does not document a chat channel,
response-time guarantee, or paid support program.

| Need | First action |
|---|---|
| install, data, launch, training, checkpoint, or evaluation failure | search {doc}`Troubleshooting <../operations/troubleshooting>`, then search GitHub issues |
| reproducible software bug | open a [GitHub issue](https://github.com/DeepLearnPhysics/particle-imaging-models/issues) with the minimal reproducer below |
| documentation error | open an issue and link the page/heading plus the code or command that disagrees |
| feature or research-method proposal | open an issue describing the user outcome, scientific contract, compatibility impact, and bounded validation plan |
| private data/access/account problem | contact the owner of that dataset or computing site; do not post credentials or private paths publicly |
| question about a published model/result | include the exact Hub repository/revision, config, dataset revision, and metric definition |

## Before opening an issue

```bash
git rev-parse HEAD
git status --short
uv lock --check
uv run pimm launch --train.config <config> --dry-run
```

Search the exact first error, not only the final launcher exit. Reduce the run
to one process, a bounded event subset, and the smallest config change that
still fails when that is scientifically safe.

## Minimal bug report

Copy this checklist into the issue:

```markdown
### Expected behavior
[What should have happened?]

### Actual behavior
[First error and relevant preceding log lines]

### Reproducer
[Exact command and smallest config/data shape summary]

### Environment
- pimm version and full Git commit:
- modified worktree: yes/no (attach relevant diff)
- OS, Python, PyTorch, CUDA runtime, NVIDIA driver:
- GPU model/count; world size and workers:
- install path: locked source / standard image / NERSC image / other:

### Inputs
- config or resolved_config.json:
- checkpoint URI + immutable revision/checksum:
- dataset type, revision, split, and minimal schema/shapes:

### Distributed context (if applicable)
- rank/node/job ID of first failure:
- scheduler output and rendezvous/NCCL error:
```

Redact tokens, credentials, private hostnames, usernames, storage paths, and
unreleased data. Do not attach a full dataset when a schema, synthetic fixture,
or a few authorized events reproduce the failure.

## Scientific questions need provenance

“My metric differs” is not actionable without the checkpoint revision,
resolved preprocessing, dataset revision/split, class map, ignored labels,
metric aggregation, and evaluation code revision. Compare those before tuning
the model. The checklist in {doc}`Evaluate <../workflows/evaluate>` is the
minimum useful context.

## Security and private reports

The repository does **not currently include `SECURITY.md` or a documented
private vulnerability-reporting address**. Do not put credentials, exploitable
details involving private infrastructure, or sensitive data in a public issue.

:::{admonition} TODO
:class: pimm-todo
Publish a security policy with supported versions and a private reporting
route. Until then, contributors must obtain an appropriate private contact from
the DeepLearnPhysics/project maintainers before sharing sensitive details.
:::

## Contribution path

If the fix is understood and scoped, follow {doc}`Contributing
<../extend/contributing>`. Opening a pull request does not require a preceding
issue for a small documentation or test fix, but larger behavior/scientific
changes benefit from agreement on the contract first.
