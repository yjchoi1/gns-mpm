#for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25 26
#do
# 	export MPM_DIR="./mpm/mpm-small-train${i}/"

#	python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/results/2d-sand-column" --dt=1.0 --output="${MPM_DIR}/train-s${i}.npz"
#done



export MPM_DIR="./mpm"
export DATA_NAME="mpm-small-test6-2"


python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/${DATA_NAME}/results/2d-sand-column" --dt=1.0 --output="${MPM_DIR}/${DATA_NAME}/${DATA_NAME}.npz"
