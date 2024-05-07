module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python3 -m virtualenv venv-frontera-cpu
source venv-frontera-cpu/bin/activate

which python
python -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
python -m pip install -r requirements.txt


# test env
# --------

echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python gns/test/test_pytorch.py 

echo 'test_torch_geometric.py -> no retun if import sucessful'
python gns/test/test_torch_geometric.py

# Clean up
# --------
#deactivate
#rm -r venv
