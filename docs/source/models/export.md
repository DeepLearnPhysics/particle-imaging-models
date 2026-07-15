# Export and publish a model

A training checkpoint is for continuing an experiment. A **model export** is
for inference, fine-tuning, archiving, or sharing. Exporting keeps the model
weights and, when available, the construction config; it intentionally leaves
trainer state behind.

| Need | Use |
|---|---|
| Continue an interrupted run | Resume the training checkpoint |
| Load a model with one Python call | Export, then use {py:func}`~pimm.from_pretrained` |
| Share through Hugging Face | Export and push the export directory |
| Publish only part of a checkpoint | Remap/filter a state dict before export |

## What is written

```text
artifacts/my-model/
├── model.safetensors   # weights; model.bin only when explicitly requested
├── config.json         # present when a config can be supplied or inferred
└── README.md           # present when a model card is supplied
```

Only the weights are unconditional. A config is written when it is passed to
the exporter or recoverable from the run around a checkpoint. A model card is
written only when supplied.

The export does **not** contain optimizer, scheduler, gradient-scaler,
dataloader, RNG, epoch, or distributed trainer state. It cannot resume a run.

:::{warning}
Export into a new or empty directory. {py:func}`~pimm.save_pretrained` does not
clear existing
files, and the loader prefers `model.safetensors` over `model.bin`. Reusing a
directory after changing formats can therefore leave a stale weight file with
higher priority.
:::

## Export a run from the CLI

Preview path resolution first:

```bash
uv run pimm export --run-dir exp/panda/semseg/my-run --dry-run
```

The default checkpoint name is `last`. Export it to a portable directory:

```bash
uv run pimm export \
  --run-dir exp/panda/semseg/my-run \
  last \
  artifacts/panda-semantic-my-run
```

With `--run-dir`, pimm looks under `<run-dir>/model/` for a split checkpoint
directory, `.pth`, `.safetensors`, or `.bin`. It prefers the run's
`resolved_config.json` as export provenance and falls back to `config.py`.

You can instead provide a checkpoint path and config explicitly:

```bash
uv run pimm export \
  exp/panda/semseg/my-run/model/model_best.pth \
  artifacts/panda-semantic-best \
  --config exp/panda/semseg/my-run/config.py
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Print resolved checkpoint, config, output, format, and device without writing |
| `--device cuda:0` | Load/consolidate tensors on that device; default is CPU |
| `--model-card path/to/README.md` | Include the file as the export's model card |
| `--no-safe-serialization` | Write `model.bin` with `torch.save` instead of safetensors |
| `--push-to-hub org/name` | Upload after a successful export; the CLI creates a private repo by default |
| `--public` | With `--push-to-hub`, create a public repository |

Run `pimm export --help` for the complete parser-generated reference.

## Export from Python

{py:func}`~pimm.save_pretrained` accepts a `torch.nn.Module`, a state-dict-like
mapping, a training checkpoint mapping, or a checkpoint path.

```python
import pimm

export_dir = pimm.save_pretrained(
    model,
    "artifacts/my-model",
    cfg=cfg,
    safe_serialization=True,
    model_card=model_card_markdown,
)
```

For a checkpoint path, the run config can often be inferred:

```python
pimm.save_pretrained(
    "exp/panda/semseg/my-run/model/last",
    "artifacts/my-model",
)
```

The split checkpoint must contain `weights.pth`. Passing the raw
`trainer.dcp/` directory raises an error because it is distributed trainer
state, not a consolidated model weight.

### Which config is recorded?

The first available source wins:

1. `training_config={...}`
2. `cfg=...`
3. `config_path="config.py"` or `config.json`
4. a recognized config in the run directory inferred from the checkpoint path

The file is always named `config.json`. It may contain a full resolved training
config or a bare model config; {py:func}`~pimm.from_pretrained` accepts either
and extracts a
top-level `model` mapping when present.

Before writing, pimm sanitizes path-like values:

- absolute and `hf://` values under load-trigger keys such as `weight`,
  `pretrained`, and `checkpoint` are set to `null`;
- other absolute-path or `hf://` strings are replaced by `<redacted>`;
- architecture and ordinary hyperparameters are retained.

This is a guardrail, not a secret scanner. Inspect `config.json` before
publishing: relative paths, free-form strings, project names, hostnames, or
credentials in unexpected fields are not guaranteed to be removed.

## Verify the export

Do this before uploading:

```python
import pimm

model, metadata = pimm.from_pretrained(
    "artifacts/my-model",
    device="cpu",
    strict=True,
    return_metadata=True,
)

print(metadata["model_config"]["type"])
print(metadata["weights"])
```

A strict round trip proves that the saved architecture can be rebuilt and that
every state-dict key matches. It does **not** validate scientific equivalence by
itself. Also run one representative transformed event through both the source
and reloaded models and compare outputs with an appropriate numerical
tolerance.

Before release, check:

- the export loads in a clean environment with the documented pimm version;
- the preprocessing recipe is versioned and linked;
- class names and order match the head;
- postprocessing thresholds are recorded for detector models;
- a held-out metric includes its split and exact definition;
- intended use, limitations, training data, license, and citations are stated.

## Publish to Hugging Face

Authenticate without putting a token in shell history:

```bash
hf auth login
```

Then export and upload in one command:

```bash
uv run pimm export \
  --run-dir exp/panda/semseg/my-run \
  last \
  artifacts/my-model \
  --model-card model-card.md \
  --push-to-hub my-org/my-model
```

The CLI creates the repository as private unless `--public` is present. To
publish an export that already exists:

```python
from pimm.export import push_to_hub

push_to_hub(
    "artifacts/my-model",
    "my-org/my-model",
    private=True,
)
```

{py:func}`~pimm.push_to_hub` uploads only recognized weights, recognized config
names, and
`README.md`. Extra analysis files in the directory are not uploaded by this
helper.

### Model card starter

Keep the first screen operational: say what the model does, what goes in, what
comes out, and how to load it. Fill every bracketed field before making the
repository public.

````markdown
# [Model name]

[One sentence: task, detector/modality, and intended use.]

```python
import pimm
model = pimm.from_pretrained("[org/repository]", device="cuda")
```

## Input contract

- Raw fields and units: [coord, energy, ...]
- Preprocessing: [link to a versioned evaluation config]
- Packed feature order: [for example, coord then energy]

## Output contract

- Model type: `[registry type]`
- Output keys and shapes: [list]
- Class order: [list, or not applicable]
- Postprocessing: [thresholds/config, or not applicable]

## Evaluation

- Dataset and split: [fill in]
- Metric and result: [fill in]
- Evaluation command/config revision: [fill in]

## Intended use and limitations

[Supported use, known domain limits, and uses that have not been validated.]

## Provenance

- pimm version/commit: [fill in]
- Training recipe: [link]
- Training data: [link]
- License and citations: [fill in]
````

## Export limitations

- An export depends on architecture code in the installed pimm version; it is
  not a self-contained executable.
- `config.json` reconstructs a model, not its transform pipeline or scientific
  interpretation.
- Safetensors stores tensors safely; `model.bin` uses Python pickle machinery
  when loaded and should only come from a trusted source.
- Hub exports are weights-only. Keep the original run checkpoint and metadata
  when reproducible continuation matters.
