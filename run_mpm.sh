#!/bin/bash
#SBATCH -J mpm-6k-train0
#SBATCH -o mpm-6k-train0.o%j
#SBATCH -e mpm-6k-train0.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p skx-normal
#SBATCH -t 8:00:00
#SBATCH -A Material-Point-Metho

module reset

export MPM_DIR="./mpm/mpm-6k-train0/"
echo "${MPM_DIR}"

/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_stress.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm-initial-vel.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume_initial_vel.json -f "${MPM_DIR}"

#~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
