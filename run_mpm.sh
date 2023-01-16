#!/bin/bash
#SBATCH -J mpm-small-test5-3
#SBATCH -o mpm-small-test5-3.o%j
#SBATCH -e mpm-small-test5-3.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 02:00:00

set -e

module reset

for i in "5-3"
do
export MPM_DIR="/work2/08264/baagee/frontera/gns-mpm/mpm/mpm-small-test${i}/"
echo "${MPM_DIR}"
#echo INPUT_FILE="./input_file.json"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_stress.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_no_initial_stress.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_vel.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume_initial_vel.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i input_file.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i input_file_resume.json -f "${MPM_DIR}"
done

#~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
