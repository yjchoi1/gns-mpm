#!/bin/bash
#SBATCH -J mpm
#SBATCH -o mpm.o%j
#SBATCH -e mpm.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 05:00:00
export MPM_DIR="./mpm/mpm-train1/"
echo "${MPM_DIR}"

~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

