module load phdf5

module load gcc/9.1.0  
module load mvapich2/2.3.7
module load intel/19.1.1
module load intel/19.1.1  impi/19.0.7
module load intel/19.1.1  impi/19.0.9
module load python3/3.9

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

source ../gns-mpm-data/venv-stempede-cpu/bin/activate

which python
