# #!/bin/bash

# #Submit this script with: sbatch thefilename

# ##################################################################
# ## SLURM Defintions
# ##################################################################
# #SBATCH --time=10:00:00                                 # walltime
# #SBATCH --nodes=1                                       # number of nodes
# #SBATCH --ntasks=1                                      # limit to one node
# #SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
# #SBATCH --partition=capella
# #SBATCH --mem-per-cpu=12G                             # memory per CPU core
# #SBATCH --gres=gpu:1                                    # number of gpus
# #SBATCH -J "SAM2"                           # job name
# #SBATCH --output=out/SAM2-%j.out
# #SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
# #SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
# #SBATCH -A p_biiax
# ##################################################################
# ##################################################################


# # load the modules
# ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0

# nvidia-smi


# # Activate virtual enviroment
# source .venv_c/bin/activate


FILELIST=(
    "data/Promotion/200/batch_02/01.tif"
    "data/Promotion/200/batch_02/02.tif"
    "data/Promotion/200/batch_02/03.tif"
    "data/Promotion/245/batch_02/01.tif"
    "data/Promotion/245/batch_02/02.tif"
    "data/Promotion/245/batch_02/03.tif"
)



for FILE in "${FILELIST[@]}"; do
    OUT="results/${FILE#data/Promotion/}"
    OUT="${OUT%/*}"   # drop the trailing sub-batch folder, e.g. /03
    echo "Processing $FILE -> $OUT ..."
    python scripts/analyse.py --input="$FILE" --output="$OUT" --save_fig --mask_aware --max_area=8000
done