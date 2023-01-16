#!/bin/bash

module reset

# start env
# ---------

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

source venv-frontera-cpu/bin/activate

# test env
# --------
echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python gns/test/test_pytorch.py 

echo 'test_torch_geometric.py -> no retun if import sucessful'
python gns/test/test_torch_geometric.py
