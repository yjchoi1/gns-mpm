MPM_DIR="./mpm/mpm-large-train13"
cd ${MPM_DIR}

python3 ../../mpm_input_gen/make_mpm_input.py --x_bounds 0.0 1.0 --y_bounds 0.0 1.0 --dx 0.01 --dy 0.01 --x_range 0.695 0.995 --y_range 0.20 0.50 --randomness 0.8

cd ../../

#mv particles.txt "${MPM_DIR}"
#mv entity_sets.json "${MPM_DIR}"
#mv mesh.txt "${MPM_DIR}"
#mv particles-stresses.txt "${MPM_DIR}"
#mv initial_config.png "${MPM_DIR}"

