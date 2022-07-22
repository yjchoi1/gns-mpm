#!/bin/bash
#SBATCH -J mpm-train8
#SBATCH -o mpm-train8.o%j
#SBATCH -e mpm-train8.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 06:00:00
export MPM_DIR="./mpm/mpm-14/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

