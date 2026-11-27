# Export and publish

Run `pimm export` to turn a run's checkpoint into a portable directory that you can load with `pimm.from_pretrained`.

```bash
uv run pimm export --run-dir exp/panda/v1/semseg/my-run --dry-run
uv run pimm export --run-dir exp/panda/v1/semseg/my-run last artifacts/my-model
```

With `--run-dir`, the first positional argument is the name of a checkpoint under `<run-dir>/model/` (`last` by default). The export contains these files:

```text
artifacts/my-model/
├── model.safetensors   weights (model.bin with --no-safe-serialization)
├── config.json         the run's config, when it can be found
└── README.md           only with --model-card
```

An export contains weights and the config only. It has no optimizer, scheduler or data-loader state, so you can't resume a run from it.

In `config.json`, absolute and `hf://` paths under weight-loading keys (`weight`, `pretrained`, `checkpoint`) become `null`, and other absolute paths become `<redacted>`. Relative paths and free-form strings are left unchanged.

Export into an empty directory. Old files there are not deleted, and when a directory has both `model.safetensors` and `model.bin`, the safetensors file is loaded.

## Export from Python

```python
import pimm

pimm.save_pretrained("exp/panda/v1/semseg/my-run/model/last", "artifacts/my-model")
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
uv run pimm export --run-dir exp/panda/v1/semseg/my-run last artifacts/my-model \
  --model-card model-card.md --push-to-hub my-org/my-model
```

The repository is private unless you add `--public`. To upload an export that already exists:

```python
import pimm

pimm.push_to_hub("artifacts/my-model", "my-org/my-model", private=True)
```

Only the weights, `config.json` and `README.md` are uploaded; nothing else in the directory is.
