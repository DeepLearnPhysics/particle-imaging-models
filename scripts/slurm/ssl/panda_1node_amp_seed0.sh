#!/bin/bash
#SBATCH --job-name=lejepa_hopper
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=240:00:00
#SBATCH --account=neutrino:ml-dev
#SBATCH --partition=hopper
#SBATCH --output=slurm_logs/%j_%n_%x_%a.txt
#SBATCH --array=5

set -e

export PYTHONFAULTHANDLER=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif
export NCCL_SOCKET_IFNAME=^docker0,lo  # use any interface except docker and loopback

# Get current date and time in format YYYY-MM-DD_HH-MM
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
# Set SSL certificates for Weights & Biases. This may not
# be necessary, but was needed for our HPC cluster.
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt     

CONFIG="pretrain-sonata-v1m1-pilarnet-smallmask"

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MAX_LEN=100
    EPOCH=100000
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    MAX_LEN=1000
    EPOCH=10000
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    MAX_LEN=10000
    EPOCH=1000
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    MAX_LEN=100000
    EPOCH=100
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    MAX_LEN=1000000
    EPOCH=100
fi

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/particle-imaging-models/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -d panda/pretrain -c ${CONFIG} -n ${CONFIG}-${MAX_LEN}-${EPOCH}-${CURRENT_DATETIME} -- --options data.train.max_len=${MAX_LEN} epoch=${EPOCH}"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"