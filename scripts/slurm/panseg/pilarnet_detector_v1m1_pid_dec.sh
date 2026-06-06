#!/bin/bash
#SBATCH --job-name=dec-pid-insseg-v1m1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=90:00:00
#SBATCH --account=neutrino:cider-nu
#SBATCH --partition=ampere
#SBATCH --output=slurm_logs/%j_%n_%x_%a.txt
#SBATCH --array=3

set -e

export PYTHONFAULTHANDLER=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

export NCCL_SOCKET_IFNAME=^docker0,lo  # Use any interface except docker and loopback

# Get current date and time in format YYYY-MM-DD_HH-MM
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Set SSL certificates for Weights & Biases. This may not
# be necessary, but was needed for our HPC cluster.
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt     

CKPT_BASE_PATH=/sdf/home/y/youngsam/sw/dune/representations/Pointcept/exp/pilarnet_datascale/
EXP="pretrain-sonata-pilarnet-1m-amp-4GPU-2025-09-07_00-46-46-seed0"
CKPT_PATH="${CKPT_BASE_PATH}/${EXP}/model/model_last.pth"

CONFIG="detector-v1m1-pt-v3m2-ft-pid-dec"

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MAX_LEN=100
    EPOCH=200000
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    MAX_LEN=1000
    EPOCH=20000
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    MAX_LEN=10000
    EPOCH=1
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    MAX_LEN=100000
    EPOCH=200
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    MAX_LEN=1000000
    EPOCH=20
fi

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/particle-imaging-models/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 2 -g 4 -c panda/panseg/${CONFIG} -w ${CKPT_PATH} -n ${CONFIG}-${MAX_LEN}-${EPOCH}-${CURRENT_DATETIME} -- --options data.train.max_len=${MAX_LEN} epoch=${EPOCH}"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"