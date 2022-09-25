python3 mpm_input_gen/make_mpm_input.py --x_bounds 0.0 1.0 --y_bounds 0.0 1.0 --x_range 0.0 0.3 --y_range 0.0 0.3

MPM_DIR="./mpm/abc"
mv particles.txt "${MPM_DIR}"
mv entity_sets.json "${MPM_DIR}"
mv mesh.txt "${MPM_DIR}"
