# Environment variables

Prefer explicit config fields for scientific settings and launch YAML for
machine policy. Environment variables are appropriate for storage locations,
credentials, and scheduler-provided process identity.

## Where values come from

`pimm launch` and `pimm submit` export the `env:` mapping from resolved launch
YAML, then call `scripts/train.sh`. Both `scripts/train.sh` and
`scripts/test.sh` source a repository-root `.env` file when present.

```bash
cp example.env .env
```

`.env` uses ordinary shell syntax, not a dotenv parser. Because it is sourced
inside the script, an assignment in `.env` replaces a same-named value already
in the environment. Keep each variable in one authoritative place and never
commit secrets. Direct Python API calls and `pimm export` do not source `.env`.

## Data locations

| Variable | Read by | Meaning |
|---|---|---|
| `PILARNET_DATA_ROOT_V1` | {py:class}`~pimm.datasets.pilarnet.h5.PILArNetH5Dataset` | root of revision-v1 HDF5 split directories |
| `PILARNET_DATA_ROOT_V2` | {py:class}`~pimm.datasets.pilarnet.h5.PILArNetH5Dataset` | root of revision-v2 HDF5 split directories |
| `PILARNET_DATA_ROOT_V3` | {py:class}`~pimm.datasets.pilarnet.h5.PILArNetH5Dataset` | separately supplied revision-v3 root; the standard downloader does not provide it |
| `PILARNET_PARQUET_ROOT_<REV>` | PILArNet Parquet map/stream readers | local Parquet root for the selected revision; wins over the configured Hub repository |
| `JAXTPC_DATA_ROOT` | committed JAXTPC base config | JAXTPC production root; otherwise the config retains a non-runnable placeholder path |

An explicit dataset `data_root` takes priority over the corresponding PILArNet
environment variable. HDF5 then falls back to
`~/.cache/pimm/pilarnet/<revision>` when that directory exists.

## Checkpoints and model downloads

| Variable | Meaning and precedence |
|---|---|
| `MODEL_DIR` | physical checkpoint root used by `scripts/train.sh`; the experiment's `model/` becomes a symlink. Also supplies the fallback Hub cache at `$MODEL_DIR/hub` |
| `HF_HUB_CACHE` | cache used for downloaded Hugging Face datasets and models; set this when the cache must live on a shared or high-capacity filesystem |
| `HF_HOME` | parent directory for Hugging Face state when `HF_HUB_CACHE` is not set |
| `HF_TOKEN` | standard Hugging Face authentication token for private downloads/uploads |
| `HF_HUB_DISABLE_XET` | standard Hub transfer switch; NERSC profiles set it to `1` for site connectivity |

Hub cache selection is:

```text
HF_HUB_CACHE → HF_HOME → MODEL_DIR/hub → Hugging Face default
```

Reference a Hub model by its repository URI:

```text
hf://ORG/REPOSITORY
```

## Logging and credentials

| Variable | Meaning |
|---|---|
| `WANDB_API_KEY` | non-interactive Weights & Biases authentication |
| `WANDB_MODE` | standard W&B mode such as `offline` or `disabled`; read by W&B, not interpreted by pimm |

`--run.wandb-api-key` is converted to `WANDB_API_KEY`, but putting a credential
on a command line can expose it through shell history or process listings.
Prefer `wandb login`, a protected scheduler secret, or a permission-restricted
`.env` file.

`--token` on `pimm export` has the same exposure risk. Prefer `hf auth login`
or `HF_TOKEN`.

## Run and distributed execution

| Variable | Owner | Guidance |
|---|---|---|
| `EXP_ROOT` | `scripts/train.sh` | low-level experiment root; launcher users should set `--paths.exp-root` |
| `CUDA_VISIBLE_DEVICES` | CUDA/PyTorch | constrains devices visible to local `auto` detection and rank mapping |
| `MASTER_ADDR`, `MASTER_PORT` | launcher/torchrun | rendezvous endpoint; normally rendered from job/run context or `--rdzv.endpoint` |
| `PIMM_RDZV_ID`, `PIMM_RDZV_BACKEND` | launcher/`scripts/train.sh` | advanced torchrun rendezvous overrides; normally use `--rdzv.id` and `--rdzv.backend` |
| `PIMM_NODE_RANK` | `scripts/train.sh` | explicit node rank fallback; scheduler/torchrun normally supplies rank identity |
| `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK` | torchrun | process identity consumed by pimm; do not hand-set for ordinary launches |
| `SLURM_*` | Slurm | node/task/job identity and nodelist used to derive rank and rendezvous defaults |
| `PIMM_SUBMITIT_ATTEMPT` | pimm Submitit wrapper | internal requeue-attempt number |

For cluster-specific NCCL, HDF5, certificate, module, and network settings, use
`launch/sites/<site>.yaml`. The bundled profiles show examples such as
`NCCL_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`, `HDF5_USE_FILE_LOCKING`,
`REQUESTS_CA_BUNDLE`, and `SSL_CERT_FILE`; copy only settings validated for your
site.

## Reproducibility and diagnostics

| Variable | Meaning |
|---|---|
| `CUBLAS_WORKSPACE_CONFIG` | set by pimm to `:4096:8` when deterministic setup is requested |
| `PYTHONHASHSEED` | set from the resolved seed during pimm process setup; starting Python has already initialized some hash state, so record the full environment rather than treating this as a bitwise guarantee |

## Installer-only controls

`install.sh` recognizes `PIMM_REPO`, `PIMM_BRANCH`, and `SKIP_CLONE=1` for
bootstrap/fork workflows. They affect checkout selection, not a running pimm
experiment.

## Inspect without printing secrets

```bash
env | cut -d= -f1 | sort | rg \
  '^(PIMM|PILARNET|JAXTPC|MODEL_DIR|HF_|WANDB|CUDA_VISIBLE|MASTER_|SLURM_)'

uv run pimm launch --train.config tests/tiny_semseg --dry-run
```

Rendered dry runs redact mapping keys that look like tokens, API keys,
passwords, secrets, or credentials. That is a convenience, not a complete
secret scanner; review output before sharing it.
