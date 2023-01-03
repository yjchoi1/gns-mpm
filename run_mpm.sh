#!/bin/bash
#SBATCH -J 3dsand_test0-4
#SBATCH -o 3dsand_test0-4.o%j
#SBATCH -e 3dsand_test0-4.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 24:00:00

set -e

module reset

for i in 0 1 2 3 4
do
export MPM_DIR="/work2/08264/baagee/frontera/gns-mpm/mpm/3dsand_test${i}/"
echo "${MPM_DIR}"
#echo INPUT_FILE="./input_file.json"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_stress.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_no_initial_stress.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_vel.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume_initial_vel.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i input_file.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i input_file_resume.json -f "${MPM_DIR}"
done

#~/Documents/mpm/build/mpm -i /mpm.json -f "${MPM_DIR}"
