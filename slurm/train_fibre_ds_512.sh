#!/bin/bash
# Fine-tune SAM 2.1 Hiera-Tiny on data/fibre_ds_512
# Submit from the repo root with: sbatch slurm/train_fibre_ds_512.sh

##################################################################
## SLURM Definitions
##################################################################
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH -J "SAM2-train-t-fibre512"
#SBATCH --output=out/SAM2-train-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH -A p_haftfaeden
##################################################################

set -euo pipefail

PROJECT_DIR="$HOME/projects/alpha-capella/segment-anything-2"
CONFIG="configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512.yaml"

cd "$PROJECT_DIR"
mkdir -p out

ml release/25.06 GCC/13.3.0 Python/3.12.3 OpenMPI/5.0.3 CUDA/13.0.0
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Python: $(command -v python)"
nvidia-smi
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python -c "import os; print('CPUs visible:', len(os.sched_getaffinity(0)))"

# Trainer runs locally on the allocated GPU (no submitit re-submission).
# Logs/checkpoints go to sam2_logs/$CONFIG/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run as a job step so the process is bound to all allocated CPUs
# (without srun the dataloader only saw 1 CPU).
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python training/train.py \
    -c "$CONFIG" \
    --use-cluster 0 \
    --num-gpus 1
