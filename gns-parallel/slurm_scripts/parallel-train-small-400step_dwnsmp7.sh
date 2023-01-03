#!/bin/bash

#SBATCH -J pyt_sand-small-ds-7         # Job name
#SBATCH -o pyt_sand-small-ds-7.o%j     # Name of stdout output file
#SBATCH -e pyt_sand-small-ds-7.e%j     # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu

# fail on error
set -e

# start in gns folder
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
export DATASET_NAME="sand-small-r300_parallel_dwnsmp7"
export WORK_DIR="../gns-data"

torchrun gns/train.py --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" --nsave_steps=10000 --ntraining_steps=10000000 --loss_save_freq 2
