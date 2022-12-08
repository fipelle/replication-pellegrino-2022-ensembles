#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=m64c512g
#SBATCH --job-name=$(job_name)
#SBATCH --error=$(slurm_logs_folder)/%x_%N_%j.err
#SBATCH --output=$(slurm_logs_folder)/%x_%N_%j.out
#SBATCH --chdir="../"

module add apps/julia/1.6.7

