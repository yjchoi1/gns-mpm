#!/bin/bash
#SBATCH -J mpm-train3
#SBATCH -o mpm-train3.o%j
#SBATCH -e mpm-train3.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 07:00:00

/work/08264/baagee/frontera/mpm/build/mpm -i mpm.json -f ./

