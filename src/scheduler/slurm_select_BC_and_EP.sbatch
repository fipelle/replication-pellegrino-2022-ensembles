#!/bin/bash

#SBATCH --partition=xlarge
#SBATCH --nodes=1
#SBATCH --cpus-per-task 64
#SBATCH --job-name=select_BC_and_EP
#SBATCH --error=./logs/%x_%N_%j.err
#SBATCH --output=./logs/%x_%N_%j.out
#SBATCH --chdir="../"

module add apps/julia/1.6.7
julia -p64 macro_selection.jl true 4 "./BC_and_EP_output"
