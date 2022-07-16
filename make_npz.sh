export MPM_DIR="./mpm/mpm-train1/"

python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/results/2d-sand-column" --dt=1.0 --output="test.npz"

mv test.npz gns-data/datasets/sand-2d
