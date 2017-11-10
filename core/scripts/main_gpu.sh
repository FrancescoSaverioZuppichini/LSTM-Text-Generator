#!/bin/bash -l
#
#SBATCH --job-name="dif_equat_20d"
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --error=a.%j.err
module load python/3.5.0
module load cudnn/8.0
srun python3 train.py 