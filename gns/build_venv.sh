#!/bin/bash

# make venv and activate it
virtualenv ~/python_envs/venv-gns-cpu
source ~/python_envs/venv-gns-cpu/bin/activate

# install requirements
which pip3
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
export TORCH="1.11"
export TORCH="cpu"
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip3 install -r requirements.txt

# test env
# --------

echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python test/test_pytorch.py 

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no retun if import sucessful'
python test/test_torch_geometric.py

# Clean up
# --------
#deactivate
#rm -r venv

