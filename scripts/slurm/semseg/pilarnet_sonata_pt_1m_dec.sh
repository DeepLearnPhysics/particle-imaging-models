#!/bin/bash
#SBATCH --job-name=1m_dec
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=60:00:00
#SBATCH --account=mli:nu-ml-dev
#SBATCH --partition=ampere
#SBATCH --output=slurm_logs/%j_%n_%x_%a.txt
#SBATCH --array=1

set -e
. /sdf/home/y/youngsam/sw/dune/representations/particle-imaging-models/.env
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

# define configs based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-dec-100"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-dec-1000"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-dec-10000"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-dec-100000"
fi
# do not include 1M becase that's in pilarnet_sonata_pt_ft_same_fft.sh

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/particle-imaging-models/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -c panda/semseg/${CONFIG} -n ${CONFIG}-${CURRENT_DATETIME}-seed0 -w ${CKPT_PATH}"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"