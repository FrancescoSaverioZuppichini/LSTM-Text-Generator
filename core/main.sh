#!/bin/bash -l
#
#SBATCH --job-name="lstm"
#SBATCH --partition=tflow
#SBATCH --time=00:30:00
#SBATCH --output=a.%j.out
#SBATCH --error=a.%j.err
module load python/3.5.0
module load cudnn/8.0
srun python3 train_2.py
