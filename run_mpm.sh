#!/bin/bash
#SBATCH -J train-large31
#SBATCH -o train-large31.o%j
#SBATCH -e train-large31.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 12:00:00

module reset

export MPM_DIR="./mpm/mpm-large-train31/"
echo "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm-initial.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
#~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
