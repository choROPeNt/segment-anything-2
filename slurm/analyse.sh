#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=10:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G                             # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "SAM2"                           # job name
#SBATCH --output=slurm_out/SAM2-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################







ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0

DIRLIST=(
    "/data/horse/ws/dchristi-3dseg/data/micro/160/00_series/00"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/01_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/01_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/01_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/02_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/02_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/02_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/02_series/04"
    "/data/horse/ws/dchristi-3dseg/data/micro/160/02_series/05"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/00_series/00"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/01_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/01_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/01_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/02_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/02_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/02_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/02_series/04"
    "/data/horse/ws/dchristi-3dseg/data/micro/200/02_series/05"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/00_series"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/01_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/01_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/01_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/02_series/01"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/02_series/02"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/02_series/03"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/02_series/04"
    "/data/horse/ws/dchristi-3dseg/data/micro/285/02_series/05"
)



for DIR in "${DIRLIST[@]}"; do
    echo "Processing $DIR ..."
    python scripts/analyse.py --input="$DIR" --save_fig --max_area=8000
done