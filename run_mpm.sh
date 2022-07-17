#!/bin/bash
#SBATCH -J mpm-4
#SBATCH -o mpm-4.o%j
#SBATCH -e mpm-4.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 07:00:00
export MPM_DIR="./mpm/mpm-4/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

