#!/bin/bash

#Submit this script with: sbatch slurm/benchmark_fiber_dataset_e60.sh
# Full-micrograph SAM 2 detections on the composites fibre-detection benchmark
# (Yuksel et al. 2026, data/Benchmark_fiber-dataset), then the evaluation against the
# expert reference and the 11 participants. ~12 h on an M1 Pro (MPS), far less on one GPU.

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=08:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G                               # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "SAM2-fibre-benchmark"                       # job name
#SBATCH --output=out/SAM2-fibre-benchmark-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################

ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0
nvidia-smi
source .venv_c/bin/activate

CHECKPOINT="sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512_pad256_train-val_diag_e60.yaml/checkpoints/checkpoint_060.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"   # model size of the fine-tune (Hiera-T), not the training config
PRED_IOU="${PRED_IOU:-0.7}"                    # 0.88 (in-domain sweep) misses fibres in this domain
TAG="${TAG:-e60}"

python scripts/benchmark_fiber_dataset.py detect --image all --roi full \
    --checkpoint="$CHECKPOINT" --config="$CONFIG" --pred_iou_thresh="$PRED_IOU" --tag="$TAG"
python scripts/benchmark_fiber_dataset.py evaluate --checkpoint="$CHECKPOINT" --tag="$TAG"

exit 0
