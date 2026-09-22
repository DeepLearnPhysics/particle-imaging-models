#!/bin/sh
# Shared bootstrap for scripts/train.sh and the frozen-source execution adapter.
PYTHON=$1
NUM_MACHINE=$2
NUM_GPU=$3
shift 3

NODE_RANK=${PIMM_NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}
if [ "$NUM_MACHINE" = "1" ] && [ -z "${MASTER_ADDR:-}" ]; then
  exec $PYTHON -m torch.distributed.run --standalone --nproc-per-node="$NUM_GPU" "$@"
fi
RDZV_ID=${PIMM_RDZV_ID:-${SLURM_JOB_ID:-pimm}}
RDZV_BACKEND=${PIMM_RDZV_BACKEND:-c10d}
# Avoid resolving a node hostname to a non-routable IPv6 link-local address.
if [ -n "${MASTER_ADDR:-}" ]; then
  MASTER_IPV4=$(getent ahostsv4 "$MASTER_ADDR" 2>/dev/null | awk 'NR==1{print $1}')
  [ -n "$MASTER_IPV4" ] && MASTER_ADDR="$MASTER_IPV4"
fi
RDZV_ENDPOINT=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}
NODE_RANK_ARG=""
if [ "$RDZV_BACKEND" = "static" ]; then
  NODE_RANK_ARG="--node-rank=$NODE_RANK"
fi
exec $PYTHON -m torch.distributed.run \
  --nnodes="$NUM_MACHINE" \
  --nproc-per-node="$NUM_GPU" \
  $NODE_RANK_ARG \
  --rdzv-backend="$RDZV_BACKEND" \
  --rdzv-endpoint="$RDZV_ENDPOINT" \
  --rdzv-id="$RDZV_ID" \
  "$@"
