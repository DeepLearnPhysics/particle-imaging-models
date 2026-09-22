#!/bin/sh
# Training entrypoint used by both `pimm launch` and direct shell invocations.
# It snapshots code into exp/.../code by default; `-C` disables code copy.
cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

# Load .env if present
[ -f "$ROOT_DIR/.env" ] && set -a && . "$ROOT_DIR/.env" && set +a

TRAIN_CODE=train.py

# ─── Help ───────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
Usage: sh scripts/train.sh [OPTIONS] [-- --options key=val ...]

Options:
  -c CONFIG     Config path under configs/, with or without .py
                (e.g., panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask)
  -n NAME       Experiment name (default: auto-generated from CONFIG + timestamp)
  -g GPUS       GPUs per machine (default: auto-detect all available)
  -m MACHINES   Number of machines (default: 1)
  -w WEIGHT     Path to pretrained checkpoint
  -r true       Resume training from last checkpoint
  -a NAME       Override Weights & Biases run name
  -p PYTHON     Python interpreter (default: python)
  -C            Disable code copy and run directly from repo source
  -h            Show this help

Examples:
  # Single-GPU Sonata pre-training
  sh scripts/train.sh -g 1 -c panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask

  # 4-GPU with custom experiment name
  sh scripts/train.sh -g 4 -c panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask -n my_exp

  # Override config values
  sh scripts/train.sh -g 1 -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin -- --options epoch=10

  # No code copy (changes to repo take effect immediately)
  sh scripts/train.sh -C -g 1 -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin -n dev
EOF
  exit 0
}

# ─── Defaults ───────────────────────────────────────────────────
DATASET=""
CONFIG="None"
EXP_NAME=""
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"
NO_COPY=false
# Respect an external/.env MODEL_DIR (empty by default) and export it so the
# training process sees it (checkpoint redirect + HF cache location).
export MODEL_DIR="${MODEL_DIR:-}"
EXP_ROOT=${EXP_ROOT:-exp}

while getopts "p:c:n:w:g:m:r:a:Ch" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    a)
      WANDB_NAME=$OPTARG
      ;;
    C)
      NO_COPY=true
      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      echo "Run 'sh scripts/train.sh -h' for usage."
      exit 1
      ;;
  esac
done

# shift past processed options to get extra args (e.g., --options key=val)
shift $((OPTIND-1))

# ─── Normalize config reference ─────────────────────────────────
normalize_config_ref() {
  if [ "${CONFIG}" = "None" ]; then
    echo "Error: -c CONFIG is required" >&2
    echo "Run 'sh scripts/train.sh -h' for usage." >&2
    exit 1
  fi

  config_ref="${CONFIG#./}"
  repo_config_prefix="${ROOT_DIR}/configs/"
  case "$config_ref" in
    "$repo_config_prefix"*) config_ref="${config_ref#$repo_config_prefix}" ;;
  esac
  config_ref="${config_ref#configs/}"
  config_ref="${config_ref%.py}"

  case "$config_ref" in
    */*)
      DATASET=$(dirname "$config_ref")
      CONFIG=$(basename "$config_ref")
      ;;
    *)
      DATASET=""
      CONFIG=$config_ref
      ;;
  esac

  if [ -n "$DATASET" ]; then
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
  else
    CONFIG_DIR=configs/${CONFIG}.py
  fi
}

normalize_config_ref

# ─── Validate config exists ────────────────────────────────────
if [ "${CONFIG}" != "None" ] && [ ! -f "$CONFIG_DIR" ]; then
  echo "Error: Config not found: $CONFIG_DIR"
  if [ -n "$DATASET" ]; then
    PARENT_DIR="configs/${DATASET}"
  else
    PARENT_DIR="configs"
  fi
  if [ -d "$PARENT_DIR" ]; then
    echo ""
    parent_label=${PARENT_DIR#configs/}
    [ -n "$parent_label" ] || parent_label=configs
    echo "Available configs in ${parent_label}/:"
    find "$PARENT_DIR" -maxdepth 1 -name "*.py" -not -name "__*" | sort | while read f; do
      echo "  $(basename "$f" .py)"
    done
  else
    echo ""
    echo "Config directory not found: $PARENT_DIR"
    echo ""
    echo "Available directories:"
    find configs -name "*.py" -not -path "*/_base_/*" -not -name "__*" -exec dirname {} \; | \
      sort -u | sed 's|^configs/||' | sed 's/^/  /'
  fi
  exit 1
fi

# ─── Auto-generate experiment name if not provided ──────────────
if [ -z "${EXP_NAME}" ]; then
  if [ "${CONFIG}" != "None" ]; then
    # Derive the timestamp from the Slurm job start time so every node in a
    # multi-node job agrees on the exp name. Per-node `date` makes ranks pick
    # names a second apart -> the rank-0 code snapshot lands in a different dir
    # than the other ranks wait on -> snapshot-wait deadlock -> torchrun
    # rendezvous times out. Fall back to `date` for non-Slurm/local runs.
    if [ -n "${SLURM_JOB_START_TIME:-}" ]; then
      CURRENT_DATETIME=$(date -d "@${SLURM_JOB_START_TIME}" +"%Y-%m-%d_%H-%M-%S")
    else
      CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
    fi
    EXP_NAME="${CONFIG}-${CURRENT_DATETIME}"
  else
    EXP_NAME="debug"
  fi
fi

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Config group: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

EXP_ROOT=${EXP_ROOT%/}
if [ -n "$DATASET" ]; then
  EXP_DIR=${EXP_ROOT}/${DATASET}/${EXP_NAME}
else
  EXP_DIR=${EXP_ROOT}/${EXP_NAME}
fi
echo "Experiment dir: $EXP_DIR"

# Build MODEL_SAVE_DIR and symlink if MODEL_DIR is set
if [ -n "$MODEL_DIR" ]; then
  # If MODEL_DIR is set, checkpoints go to MODEL_DIR/.../model
  MODEL_SAVE_DIR=${MODEL_DIR%/}/$EXP_DIR/model
  MODEL_LINK_DIR=${EXP_DIR}/model
  echo "MODEL_SAVE_DIR: $MODEL_SAVE_DIR"
else
  # If not set, checkpoints go to EXP_DIR/model
  MODEL_SAVE_DIR=${EXP_DIR}/model
  MODEL_LINK_DIR=""
fi

if [ "${RESUME}" = true ]
then
  if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: resume=true but experiment directory does not exist: $EXP_DIR" >&2
    exit 2
  fi
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$($PYTHON -m pimm.utils.path latest-checkpoint "${EXP_DIR}/model")
  if [ -z "$WEIGHT" ]; then
    echo "ERROR: resume=true but no complete checkpoint found at ${EXP_DIR}/model/last, last.prev, or model_last.pth" >&2
    exit 2
  fi
fi

# Ensure the physical checkpoint dir exists and expose it at EXP_DIR/model.
# This must happen regardless of the code-copy mode below; the link is only
# created when EXP_DIR/model is absent or already a link, so runs that saved
# checkpoints into a plain EXP_DIR/model directory keep working.
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  mkdir -p "$MODEL_SAVE_DIR"
  if [ -n "$MODEL_LINK_DIR" ] && { [ ! -e "$MODEL_LINK_DIR" ] || [ -L "$MODEL_LINK_DIR" ]; }; then
    mkdir -p "$EXP_DIR"
    ln -sfn "$(realpath "$MODEL_SAVE_DIR")" "$MODEL_LINK_DIR"
  fi
fi

# ─── Code snapshot vs no-code-copy ──────────────────────────────
if [ "$NO_COPY" = true ]; then
  CODE_DIR="."
  export PYTHONPATH=.
  echo "No code copy: running from repo source"
elif [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]; then
  CODE_DIR=${EXP_DIR}/code
  export PYTHONPATH=$CODE_DIR
  echo "Resuming: running from codebase snapshot $CODE_DIR"
else
  RESUME=false
  CODE_DIR=${EXP_DIR}/code
  OUTER_RANK=${SLURM_PROCID:-0}
  SNAPSHOT_DONE="$CODE_DIR/.pimm_snapshot_complete_${SLURM_JOB_ID:-local}"

  if [ "$OUTER_RANK" = "0" ]; then
    mkdir -p "$CODE_DIR"

    echo " =========> CREATE EXP DIR <========="
    echo "Experiment dir: $EXP_DIR"
    cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null
    touch "$SNAPSHOT_DONE"
  else
    for _ in $(seq 1 600); do
      [ -f "$SNAPSHOT_DONE" ] && break
      sleep 1
    done
    if [ ! -f "$SNAPSHOT_DONE" ]; then
      echo "ERROR: timed out waiting for code snapshot: $SNAPSHOT_DONE" >&2
      exit 2
    fi
  fi

  export PYTHONPATH=$CODE_DIR
  echo "Running from repo snapshot: $CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR

sleep 0.5

echo " =========> RUN TASK <========="
ulimit -n 65536

run_python() {
  # Preserve each override as one argument, including strings and lists with spaces.
  if [ "${WEIGHT}" != "None" ]; then
    set -- "resume=$RESUME" "weight=$WEIGHT" "$@"
  fi
  if [ -n "$WANDB_NAME" ]; then
    set -- "wandb_run_name=$WANDB_NAME" "$@"
  fi
  set -- --config-file "$CONFIG_DIR" --options "save_path=$EXP_DIR" "$@"

  exec sh "$ROOT_DIR/pimm/launch/torchrun.sh" "$PYTHON" "$NUM_MACHINE" "$NUM_GPU" \
    "$CODE_DIR/pimm/$TRAIN_CODE" "$@"
}

run_python "$@"
