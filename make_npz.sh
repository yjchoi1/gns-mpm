set -e

## workdir
#export MPM_DIR="./mpm"
#export DATA_CASE="mpm-large-train"
#export DATA_TAG="8"
#
#python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" --dt=1.0 \
# --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"

# multiple
#for ((DATA_TAG=0; DATA_TAG<60; DATA_TAG+=1))
for DATA_TAG in "8"
  do
    # workdir
    export MPM_DIR="./mpm"
    export DATA_CASE="mpm-small-test"
    export DATA_TAG=${DATA_TAG}
    python3 make_npz/convert_hdf5_to_npz.py \
    --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column/" \
    --dt=1.0 \
    --ndim=2 \
    --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"
#    python3 utils/animation_from_h5.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" \
#     --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results" \
#     --xboundary 0.0 1.0 --yboundary 0.0 1.0
  done