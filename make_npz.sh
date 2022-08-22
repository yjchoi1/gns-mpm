for i in 9 10 11 12 13 14 15 16 17 18
do
	export MPM_DIR="./mpm/mpm-train${i}/"

	python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/results/2d-sand-column" --dt=1.0 --output="${MPM_DIR}/train-${i}.npz"
done

# export MPM_DIR="./mpm/mpm-test1/"

# python3 make_npz/convert_hdf5_to_npz.py --path="${MPM_DIR}/results/2d-sand-column" --dt=1.0 --output="${MPM_DIR}/test-mpm-test1.npz"
