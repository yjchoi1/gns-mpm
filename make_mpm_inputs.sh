python3 mpm_input_gen/make_mpm_input.py --x_bounds 0.0 2.0 --y_bounds 0.0 2.0 --x_range 0.8 1.2 --y_range 0.0 0.6

MPM_DIR="./mpm/mpm-small-test7"
mv particles.txt "${MPM_DIR}"
mv entity_sets.json "${MPM_DIR}"
mv mesh.txt "${MPM_DIR}"
