#!/bin/bash
#SBATCH --job-name=dec_data_scaling
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=60:00:00
#SBATCH --account=mli:nu-ml-dev
#SBATCH --partition=ampere
#SBATCH --output=slurm_logs/%j_%n_%x_%a.txt
#SBATCH --array=1-5

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

CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-dec"

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MAX_LEN=100
    EPOCH=200000
    CKPT_EXP="pretrain-sonata-pilarnet-100-amp-4GPU-2025-09-07_01-00-46-seed0"
    WHICH="best"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    MAX_LEN=1000
    EPOCH=20000
    CKPT_EXP="pretrain-sonata-pilarnet-1k-amp-4GPU-2025-09-07_01-00-46-seed0"
    WHICH="best"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    MAX_LEN=10000
    EPOCH=2000
    WHICH="last"
    CKPT_EXP="pretrain-sonata-pilarnet-10k-amp-4GPU-2025-09-07_13-58-34-seed0"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    MAX_LEN=100000
    EPOCH=200
    CKPT_EXP="pretrain-sonata-pilarnet-100k-amp-4GPU-2025-09-07_23-24-59-seed0"
    WHICH="last"
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    MAX_LEN=1000000
    EPOCH=20
    CKPT_EXP="pretrain-sonata-pilarnet-1m-amp-4GPU-2025-09-07_00-46-46-seed0"
    WHICH="last"
fi
CKPT_PATH="${CKPT_BASE_PATH}/${CKPT_EXP}/model/model_${WHICH}.pth"

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/particle-imaging-models/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -c panda/semseg/${CONFIG} -w ${CKPT_PATH} -n ${CONFIG}-${MAX_LEN}-${EPOCH}-${CURRENT_DATETIME} -- --options data.train.max_len=${MAX_LEN} epoch=${EPOCH}"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"
