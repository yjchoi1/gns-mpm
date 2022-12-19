MPM_DIR="./mpm/mpm-small-test5-2"
cd ${MPM_DIR}

python3 ../../mpm_input_gen/make_mpm_input.py \
--x_bounds 0.0 1.0 --y_bounds 0.0 1.0 \
--dx 0.0125 --dy 0.0125 \
--x_range 0.018 0.193 --y_range 0.0 0.35 \
--randomness 0.8 \
--k0 0.5 --density 1800.0
#--initial_vel -1.0 0.0 \







cd ../../

#mv particles.txt "${MPM_DIR}"
#mv entity_sets.json "${MPM_DIR}"
#mv mesh.txt "${MPM_DIR}"
#mv particles-stresses.txt "${MPM_DIR}"
#mv initial_config.png "${MPM_DIR}"

