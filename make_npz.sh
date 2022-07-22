for i in 0 1 2 3 4 5 6 7 8
do
	export MPM_DIR="./mpm/mpm-train${i}/"

	python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/results/2d-sand-column" --dt=1.0 --output="${MPM_DIR}/train-${i}.npz"
done
