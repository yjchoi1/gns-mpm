#!/bin/bash
#SBATCH -J mpm-train22
#SBATCH -o mpm-train22.o%j
#SBATCH -e mpm-train22.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 08:00:00
#SBATCH -A ECS22003

export MPM_DIR="./mpm/mpm-train22/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"

