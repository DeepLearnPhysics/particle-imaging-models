#!/bin/bash
# Multi-node training template (Apptainer).
# sbatch scripts/slurm/multinode.slurm.sbatch

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=myaccount
#SBATCH --output=slurm_logs/%j_%x.txt

set -e
mkdir -p slurm_logs
export NCCL_SOCKET_IFNAME=^docker0,lo

# Uncomment for SLAC S3DF:
# export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# BIND="/sdf,/fs,/lscratch"

# ── Edit these ──────────────────────────────────────────────────
SIF=pimm.sif
DATASET="panda/pretrain"
CONFIG="pretrain-sonata-v1m1-pilarnet-smallmask"
EXTRA=""  # e.g. "-- --options epoch=100 data.train.max_len=100000"
BIND="${BIND:-}"
# ────────────────────────────────────────────────────────────────

CMD="sh scripts/train.sh -m $SLURM_NNODES -g $SLURM_GPUS_ON_NODE -c $DATASET/$CONFIG $EXTRA"
apptainer exec --nv ${BIND:+--bind $BIND} $SIF bash -c "$CMD"
