set -e

## workdir
#export MPM_DIR="./mpm"
#export DATA_CASE="mpm-large-train"
#export DATA_TAG="8"
#
#python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" --dt=1.0 \
# --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"

# multiple
#for ((DATA_TAG=0; DATA_TAG<30; DATA_TAG+=1))
for DATA_TAG in "1-1" "2-1" "3" "4" "0-3" "5-1" "6-1"
  do
    # workdir
    export MPM_DIR="./mpm"
    export DATA_CASE="mpm-small-test"
    export DATA_TAG=${DATA_TAG}
    python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" --dt=1.0 \
     --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"
  done