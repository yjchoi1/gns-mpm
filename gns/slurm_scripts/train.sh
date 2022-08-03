#!/bin/bash

#SBATCH -J pyt_sand3d_train         # Job name
#SBATCH -o pyt_sand3d_train.o%j     # Name of stdout output file
#SBATCH -e pyt_sand3d_train.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=jvantassel@tacc.utexas.edu
#SBATCH -A OTH21021          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
source start_venv.sh
	
# assume data is already downloaded and hardcode WaterDropSample

export DATASET_NAME="sand-2d"
export WORK_DIR="../gns-data"

python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}" \
--model_path="${WORK_DIR}/models/${DATASET_NAME}" \
--output_path=""${WORK_DIR}/rollouts/${DATASET_NAME}" \
--nsave_steps=100 \
--ntraining_steps=300 \
# --model_file="latest" \
# --train_state_file="latest"
