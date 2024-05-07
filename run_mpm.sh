#!/bin/bash
#SBATCH -J sand2d-extrapol
#SBATCH -o sand2d-extrapol.o%j
#SBATCH -e sand2d-extrapol.e%j
#SBATCH -A BCS20003
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p small
#SBATCH --mail-type=all
#SBATCH --mail-user=yj.choi@utexas.edu
#SBATCH -t 6:00:00

# Currently, mpm runs only in cpu node in frontera due to default library setting in TACC (7/22/2023)
module reset
module load intel
module load libfabric

for i in 1
do
MPM_DIR="/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions_extrapolation/extrapol${i}/"
echo "${MPM_DIR}"
#echo INPUT_FILE="./input_file.json"

#timeout 30s /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "${MPM_DIR}"
/work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "${MPM_DIR}"
# /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input_resume.json -f "${MPM_DIR}"

done
