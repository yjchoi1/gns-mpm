#!/bin/bash
#SBATCH -J train-large9
#SBATCH -o train-large9.o%j
#SBATCH -e train-large9.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 03:00:00

module reset

export MPM_DIR="./mpm/mpm-large-train9/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
# ~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
