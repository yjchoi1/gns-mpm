#!/bin/bash

#SBATCH -J pyt_sand-small-r300-parallel_ds         # Job name
#SBATCH -o pyt_sand-small-r300-parallel_ds.o%j     # Name of stdout output file
#SBATCH -e pyt_sand-small-r300-parallel_ds.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A OTH21021          # Allocation
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu

# fail on error
set -e

source start_venv.sh

# directories
export DATASET_NAME="sand-small-r300-400step_parallel_dwnsmp"
export WORK_DIR="../gns-data"

# resume
torchrun gns/train.py gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" \
--model_path="${WORK_DIR}/models/${DATASET_NAME}/" \
--output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" \
--nsave_steps=10000 --ntraining_steps=10000000 --loss_save_freq=2 \
--model_file="latest" \
--train_state_file="latest"