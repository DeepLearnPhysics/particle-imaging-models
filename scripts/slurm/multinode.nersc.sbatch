#!/bin/bash
# Multi-node training on NERSC Perlmutter (Shifter).
# Pull the image first: shifterimg pull youngsm/pimm-nersc:pytorch2.5.0-cuda12.4

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --account=<your_nersc_account>
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --image=youngsm/pimm-nersc:pytorch2.5.0-cuda12.4
#SBATCH --module=gpu,nccl-plugin
#SBATCH --output=slurm_logs/%j_%x.txt

set -e
mkdir -p slurm_logs
export PYTHONFAULTHANDLER=1

# ── Edit these ──────────────────────────────────────────────────
DATASET="panda/pretrain"
CONFIG="pretrain-sonata-v1m1-pilarnet-smallmask"
MAX_LEN=1000000
EPOCH=100
BATCH_SIZE=256                    # total across all GPUs
EXTRA=""                          # additional --options overrides
# ────────────────────────────────────────────────────────────────

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${CONFIG}-${MAX_LEN}-${EPOCH}ep-bs${BATCH_SIZE}-${CURRENT_DATETIME}"

if (( BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "Error: BATCH_SIZE=${BATCH_SIZE} not divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 1
fi

TRAIN_PATH=$(pwd)/scripts/train.sh

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
shifter /bin/bash -lc '
    export LD_LIBRARY_PATH=/opt/hdf5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export HDF5_USE_FILE_LOCKING=FALSE
        exec /bin/sh "'"$TRAIN_PATH"'" \
        -m "'"$NUM_NODES"'" \
        -g "'"$GPUS_PER_NODE"'" \
        -c "'"$DATASET"'/'"$CONFIG"'" \
        -n "'"$EXP_NAME"'" \
        -- --options \
            data.train.max_len="'"$MAX_LEN"'" \
            epoch="'"$EPOCH"'" \
            batch_size="'"$BATCH_SIZE"'" \
            seed=0 \
            '"$EXTRA"'
'
