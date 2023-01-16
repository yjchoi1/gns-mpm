#!/bin/bash

module reset

# start env
# ---------
ml cuda/11.3
ml cudnn
ml nccl

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

source venv-frontera-gpu/bin/activate

# test env
# --------
echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python gns/test/test_pytorch.py 

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python gns/test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no retun if import sucessful'
python gns/test/test_torch_geometric.py
