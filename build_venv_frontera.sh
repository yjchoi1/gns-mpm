ml cuda/11.3
ml cudnn
ml nccl

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python3 -m virtualenv venv-frontera
source venv-frontera/bin/activate

which python
python -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12+11.3.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12+11.3.html
pip install torch-geometric
python -m pip install -r requirements.txt


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
