MPM_DIR="../gns-mpm-data/mpm/mpm-small-test5-5"
cd ${MPM_DIR}

python3 ../../../gns-mpm/mpm_input_gen/make_mpm_input.py \
--x_bounds 0.0 1.0 --y_bounds 0.0 1.0 \
--dx 0.025 --dy 0.025 \
--x_range 0.020 0.14 --y_range 0.0 3.6 \
--randomness 0.8 \
--density 1800.0 \
#--k0 0.5
#--initial_vel -1.0 0.0 \







cd ../../

#mv particles.txt "${MPM_DIR}"
#mv entity_sets.json "${MPM_DIR}"
#mv mesh.txt "${MPM_DIR}"
#mv particles-stresses.txt "${MPM_DIR}"
#mv initial_config.png "${MPM_DIR}"

