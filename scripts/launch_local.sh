#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then echo "Usage: $0 <GPUS_PER_NODE> <CONFIG> [overrides...]"; exit 1; fi
GPUS_PER_NODE="$1"; shift
CONFIG="$1"; shift
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29400}
export NUM_NODES=${NUM_NODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE="${GPUS_PER_NODE}"
torchrun --nnodes="${NUM_NODES}" --nproc_per_node="${GPUS_PER_NODE}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" -m src.train --config "${CONFIG}" "$@"
