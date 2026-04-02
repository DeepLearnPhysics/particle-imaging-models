#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --account=<your_nersc_account>
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --image=youngsm/pimm:v1.0.0-pytorch2.5.0-cuda12.4-cudnn9-devel
#SBATCH --output=slurm_logs/%j_%x.txt


# Shifter automatically swaps the container's MPICH with Cray MPICH
# at runtime, so mpi4py and parallel h5py use the native interconnect.

set -e

export PYTHONFAULTHANDLER=1

# NCCL settings for Perlmutter's Slingshot network
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NET_GDR_LEVEL=PHB

# Disable HDF5 file locking on non-$SCRATCH filesystems (NERSC requirement)
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG="pretrain-sonata-v1m1-pilarnet-smallmask"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Adjust data scale per array task (uncomment #SBATCH --array=1-5 above if needed)
MAX_LEN=${MAX_LEN:-1000000}
EPOCH=${EPOCH:-10}

TRAIN_PATH=scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -d panda/pretrain -c ${CONFIG} -n ${CONFIG}-${MAX_LEN}-${EPOCH}-${CURRENT_DATETIME} -- --options data.train.max_len=${MAX_LEN} epoch=${EPOCH}"

# --module=gpu: enables CUDA device visibility inside the container
# Shifter bind-mounts $SCRATCH, $HOME, /global/cfs automatically
srun shifter --module=gpu \
    bash -c "${COMMAND}"
