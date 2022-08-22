#!/bin/bash
#SBATCH -J mpm-test2
#SBATCH -o mpm-test2.o%j
#SBATCH -e mpm-test2.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 06:00:00
#SBATCH -A ECS22003

export MPM_DIR="./mpm/mpm-test2/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

