#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then echo "Usage: $0 <CONFIG> [overrides...]"; exit 1; fi
CONFIG="$1"; shift
export MASTER_ADDR=${MASTER_ADDR:?need MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-29400}
export NUM_NODES=${NUM_NODES:?need NUM_NODES}
export NODE_RANK=${NODE_RANK:?need NODE_RANK}
export GPUS_PER_NODE=${GPUS_PER_NODE:?need GPUS_PER_NODE}
torchrun --nnodes="${NUM_NODES}" --nproc_per_node="${GPUS_PER_NODE}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" -m src.train --config "${CONFIG}" "$@"
