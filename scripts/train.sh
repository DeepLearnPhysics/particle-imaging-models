#!/bin/sh
cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"
MODEL_DIR=""  # User may set this externally, otherwise empty by default

while getopts "p:d:c:n:w:g:m:r:a:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
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
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

# shift past processed options to get extra args (e.g., --options key=val)
shift $((OPTIND-1))
EXTRA_ARGS="$@"

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

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

if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_SAVE_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$CODE_DIR"
  
  # Determine if this is rank 0 (master process)
  # Check SLURM_PROCID first (SLURM), then RANK (PyTorch), default to 0 if not set
  RANK=${SLURM_PROCID:-${RANK:-0}}
  
  echo " =========> CREATE EXP DIR <========="
  echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
  cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null
  
  # Ensure physical checkpoint dir exists
  mkdir -p "$MODEL_SAVE_DIR"
  
  if [ -n "$MODEL_LINK_DIR" ]; then
    # Link local 'model' folder to physical checkpoint dir
    ln -sfn "$(realpath "$MODEL_SAVE_DIR")" "$MODEL_LINK_DIR"
  fi
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"

sleep 0.5

echo " =========> RUN TASK <========="
ulimit -n 65536

# Slurm Native Mode - Script handles distributed setup automatically
COMMON_ARGS="--config-file $CONFIG_DIR --options save_path=$EXP_DIR"

if [ -n "$WANDB_NAME" ]; then
  COMMON_ARGS="$COMMON_ARGS wandb_run_name=$WANDB_NAME"
fi

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE $COMMON_ARGS $EXTRA_ARGS
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE $COMMON_ARGS resume="$RESUME" weight="$WEIGHT" $EXTRA_ARGS
fi
