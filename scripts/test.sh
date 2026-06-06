#!/bin/sh
# Evaluation entrypoint for a saved experiment snapshot under exp/.../code.

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

# Load .env if present
[ -f "$ROOT_DIR/.env" ] && set -a && . "$ROOT_DIR/.env" && set +a

TEST_CODE=test.py

DATASET=""
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

while getopts "p:c:n:w:g:m:" opt; do
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
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${CONFIG}" != "None" ]; then
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

if [ -n "$DATASET" ]; then
  EXP_DIR=exp/${DATASET}/${EXP_NAME}
else
  EXP_DIR=exp/${EXP_NAME}
fi
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=${EXP_DIR}/config.py

if [ "${CONFIG}" = "None" ]
then
    CONFIG_DIR=${EXP_DIR}/config.py
elif [ -n "$DATASET" ]
then
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
else
    CONFIG_DIR=configs/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
# export PYTHONPATH=./
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

$PYTHON -u pimm/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth
