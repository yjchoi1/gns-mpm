#!/bin/bash
#SBATCH -J train-s29
#SBATCH -o train-s29.o%j
#SBATCH -e train-s29.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 02:00:00
#SBATCH -A ECS22003

export MPM_DIR="./mpm/mpm-small-train29/"
echo "${MPM_DIR}"

# /work/08264/baagee/frontera/mpm/build/mpm -i /mpm-initial.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

