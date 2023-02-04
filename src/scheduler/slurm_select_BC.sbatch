#!/bin/bash

#SBATCH --partition=xlarge
#SBATCH --nodes=1
#SBATCH --cpus-per-task 64
#SBATCH --job-name=select_BC
#SBATCH --error=/users/pellegr5/replication-pellegrino-2022-ensembles-ridge/src/scheduler/logs/%x_%N_%j.err
#SBATCH --output=/users/pellegr5/replication-pellegrino-2022-ensembles-ridge/src/scheduler/logs/%x_%N_%j.out
#SBATCH --chdir=/users/pellegr5/replication-pellegrino-2022-ensembles-ridge/src

module add apps/julia/1.6.7
julia -p48 macro_selection.jl false 4 "./BC_output"
