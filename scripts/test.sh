#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
PYTHON=python

TEST_CODE=test.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

while getopts "p:d:c:n:w:g:m:" opt; do
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

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=${EXP_DIR}/config.py

if [ "${CONFIG}" = "None" ]
then
    CONFIG_DIR=${EXP_DIR}/config.py
else
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
# export PYTHONPATH=./
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth
