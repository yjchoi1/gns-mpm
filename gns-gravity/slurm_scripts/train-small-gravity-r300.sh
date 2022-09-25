#!/bin/bash

#SBATCH -J pyt_sand-smallg-r300         # Job name
#SBATCH -o pyt_sand-smallg-r300.o%j     # Name of stdout output file
#SBATCH -e pyt_sand-smallg-r300.e%j     # Name of stderr error file
#SBATCH -p rtx              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu

# start in slurm_scripts
source ../start_venv_frontera.sh

# assume data is already downloaded and hardcode WaterDropSample
export DATASET_NAME="sand-2d-small-gravity-r300"
export WORK_DIR="../gns-data"

python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" --nsave_steps=5000 --ntraining_steps=20000000
