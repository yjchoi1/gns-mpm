#!/bin/bash

#SBATCH -A BCS20003
#SBATCH -J pyt_sand2d-r049         # Job name
#SBATCH -o pyt_sand2d-r049.o%j     # Name of stdout output file
#SBATCH -e pyt_sand2d-r049.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu


source start_venv_ls6_torch20.sh
cd ../gns-meshnet/gns
python3 train.py
