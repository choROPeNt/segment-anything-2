#!/bin/bash

#Submit this script with: sbatch slurm/analyse_160_e60.sh

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=10:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G                               # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "SAM2-analyse"                               # job name
#SBATCH --output=out/SAM2-analyse-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################


# load the modules
ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0

nvidia-smi


# Activate virtual enviroment
source .venv_c/bin/activate


CHECKPOINT="sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512_pad256_train-val_diag_e60.yaml/checkpoints/checkpoint_060.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"   # model size of the fine-tune (Hiera-T), not the training config

FILE="data/Promotion/160/batch_02/01.tif"
MASK="data/Promotion/160/batch_02/01.mask.tif"
OUT="results/160/batch_02"

echo "Processing $FILE (mask $MASK) with $CHECKPOINT -> $OUT ..."
python scripts/analyse.py \
    --input="$FILE" \
    --mask="$MASK" \
    --output="$OUT" \
    --checkpoint="$CHECKPOINT" \
    --config="$CONFIG" \
    --save_fig \
    --max_area=8000 \
    --min_circularity=0.7 \
    --min_axis_ratio=0.55 \
    --min_bright_fraction=0.5

exit 0
