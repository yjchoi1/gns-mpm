set -e

## workdir
#export MPM_DIR="./mpm"
#export DATA_CASE="mpm-large-train"
#export DATA_TAG="8"
#
#python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" --dt=1.0 \
# --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"

dim=2

# multiple
#for ((DATA_TAG=0; DATA_TAG<60; DATA_TAG+=1))
for DATA_TAG in {180..239..1}
  do
    # workdir
    export MPM_DIR="/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/"
    export DATA_CASE="sand2d_frictions"
    export DATA_TAG=${DATA_TAG}

    python3 make_npz/convert_hdf5_to_npz.py \
    --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/sand2d_frictions/" \
    --dt=1.0 \
    --ndim=${dim} \
    --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"
  done
