#!/usr/bin/env bash
set -euo pipefail

PARTITION="capella"
ACCOUNT="p_haftfaeden"
TIME="04:00:00"
CPUS=12
MEM_PER_CPU="12G"
GPUS=1
NODES=1
NTASKS=1

PROJECT_DIR="$HOME/projects/alpha-capella/segment-anything-2"
VENV_PATH=".venv"


srun \
  -p "$PARTITION" \
  -N "$NODES" \
  -n "$NTASKS" \
  --gres="gpu:${GPUS}" \
  --gpus-per-task="$GPUS" \
  -c "$CPUS" \
  --mem-per-cpu="$MEM_PER_CPU" \
  -t "$TIME" \
  --account="$ACCOUNT" \
  --pty bash -lc "
    cd '$PROJECT_DIR'

    ml release/25.06 GCC/13.3.0 Python/3.12.3 OpenMPI/5.0.3 CUDA/13.0.0
    source '$VENV_PATH/bin/activate'

    echo '--- Capella interactive session ready ---'
    echo Host: \$(hostname)
    echo Project: \$(pwd)
    echo Python: \$(command -v python)

    echo 'CUDA:'
    nvidia-smi || true

    exec bash -i
  "