ml cuda/11.3
ml cudnn
ml nccl

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load phdf5
module load gcc/9.1.0
module load python3/3.9.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH


# Install miniconda with python 3.9
conda create -n "env_frontera_gpu"
conda activate env_frontera_gpu
# Given that CUDA version is 11.3,
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pytorch-cluster -c pyg
conda install -c anaconda absl-py 
conda install -c conda-forge numpy dm-tree matplotlib-base pyevtk



