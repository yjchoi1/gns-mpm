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
for DATA_TAG in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25 26
  do
    # workdir
    export MPM_DIR="./mpm"
    export DATA_CASE="mpm-small-train"
    export DATA_TAG=${DATA_TAG}
    python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/results/2d-sand-column" --dt=1.0 \
     --output="${MPM_DIR}/${DATA_CASE}${DATA_TAG}/${DATA_CASE}${DATA_TAG}.npz"
  done