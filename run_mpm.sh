#!/bin/bash
#SBATCH -J mpm-small-tset5-5
#SBATCH -o mpm-small-tset5-5.o%j
#SBATCH -e mpm-small-tset5-5.e%j
#SBATCH -A BCS20003
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH --mail-type=all
#SBATCH --mail-user=yj.choi@utexas.edu
#SBATCH -t 2:00:00

module reset

for i in "5-5"
do
MPM_DIR="/work2/08264/baagee/frontera/gns-mpm-data/mpm/mpm-small-test${i}/"
echo "${MPM_DIR}"
#echo INPUT_FILE="./input_file.json"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_stress.json -f "${MPM_DIR}"

timeout 30s /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_vel.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume_initial_vel.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i input_file.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i input_file_resume.json -f "${MPM_DIR}"
done
