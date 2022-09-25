#!/bin/bash
#SBATCH -J test-s6-2
#SBATCH -o test-s6-2.o%j
#SBATCH -e test-s6-2.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 02:00:00

module reset

export MPM_DIR="./mpm/mpm-small-test6-2/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
# ~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
