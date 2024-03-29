#!/bin/bash

#SBATCH -J pyt_sand-small-r190-bound         # Job name
#SBATCH -o pyt_sand-small-r190-bound.o%j     # Name of stdout output file
#SBATCH -e pyt_sand-small-r190-bound.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu

# start in slurm_scripts
source start_venv.sh

# directories
export DATASET_NAME="sand-small-r190-bound"
export WORK_DIR="../gns-data"

# resume
python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" --model_file="latest" --train_state_file="latest" --nsave_steps=50000 --ntraining_steps=5000000
