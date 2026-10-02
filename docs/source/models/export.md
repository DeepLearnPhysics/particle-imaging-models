# Export and publish

`pimm export` turns a run's checkpoint into a portable directory that `pimm.from_pretrained` loads.

```bash
uv run pimm export --run-dir exp/panda/semseg/my-run --dry-run
uv run pimm export --run-dir exp/panda/semseg/my-run last artifacts/my-model
```

With `--run-dir`, the first positional argument names a checkpoint under `<run-dir>/model/` and defaults to `last`. The export contains:

```text
artifacts/my-model/
├── model.safetensors   weights (model.bin with --no-safe-serialization)
├── config.json         the run's config, when it can be found
└── README.md           only with --model-card
```

An export holds weights and the config only. It has no optimizer, scheduler or data-loader state and can't resume a run.

Before writing `config.json`, pimm sets absolute and `hf://` paths under weight-loading keys (`weight`, `pretrained`, `checkpoint`) to `null` and replaces other absolute paths with `<redacted>`. Relative paths and free-form strings pass through unchanged.

Export into an empty directory. `save_pretrained` doesn't delete old files, and the loader prefers `model.safetensors` over `model.bin`.

## Export from Python

```python
import pimm

pimm.save_pretrained("exp/panda/semseg/my-run/model/last", "artifacts/my-model")
pimm.save_pretrained(model, "artifacts/my-model", cfg=cfg, model_card=card_text)
```

A checkpoint path must be the directory that contains `weights.pth`, not `trainer.dcp/`.

## Check an export

```python
model, meta = pimm.from_pretrained("artifacts/my-model", strict=True, return_metadata=True)
print(meta["model_config"]["type"], meta["weights"])
```

## Publish to Hugging Face

```bash
uv run hf auth login
uv run pimm export --run-dir exp/panda/semseg/my-run last artifacts/my-model \
  --model-card model-card.md --push-to-hub my-org/my-model
```

The repository is private unless you add `--public`. To upload an export that already exists:

```python
import pimm

pimm.push_to_hub("artifacts/my-model", "my-org/my-model", private=True)
```

`push_to_hub` uploads the weights, `config.json` and `README.md`, and nothing else in the directory.
