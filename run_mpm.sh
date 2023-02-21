#!/bin/bash
#SBATCH -J sand3d3
#SBATCH -o sand3d3.o%j
#SBATCH -e sand3d3.e%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH -t 06:00:00

set -e

module reset

for i in 3
do
export MPM_DIR="/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d_boundary/sand3d${i}/"
echo "${MPM_DIR}"
#echo INPUT_FILE="./input_file.json"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_stress.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "../gns-mpm-data/mpm/sand3d_boundary/sand3d1/"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_initial_vel.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume_initial_vel.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_resume.json -f "${MPM_DIR}"

#/work/08264/baagee/frontera/mpm/build/mpm -i input_file.json -f "${MPM_DIR}"
#/work/08264/baagee/frontera/mpm/build/mpm -i input_file_resume.json -f "${MPM_DIR}"
done
