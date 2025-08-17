#!/bin/bash
#SBATCH --job-name=ddp-lora
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29400}

export MASTER_ADDR MASTER_PORT
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

srun bash scripts/launch_multinode.sh configs/base.yaml
