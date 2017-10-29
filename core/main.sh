#!/bin/bash -l
#
#SBATCH --job-name="abc"
#SBATCH --partition=tflow
#SBATCH --time=00:15:00
#SBATCH --output=abc.%j.out
#SBATCH --error=abc.%j.err
module load python/3.5.0
module load cudnn/8.0
srun python3 model/RNN.py
