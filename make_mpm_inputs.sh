MPM_DIR="../gns-mpm-data/mpm/mpm-small-m2"
mkdir ${MPM_DIR}
cd ${MPM_DIR}

python3 ../../../gns-mpm-dev/mpm_input_gen/make_mpm_input.py \
--x_bounds 0.0 1.0 --y_bounds 0.0 1.0 \
--dx 0.025 --dy 0.025 \
--x_range 0.0 0.9 --y_range 0.0 0.9 \
--randomness 0.8 \
--density 1800.0
#--initial_vel -1.0 0.0
#--k0 0.5








cd ../../

#mv particles.txt "${MPM_DIR}"
#mv entity_sets.json "${MPM_DIR}"
#mv mesh.txt "${MPM_DIR}"
#mv particles-stresses.txt "${MPM_DIR}"
#mv initial_config.png "${MPM_DIR}"

