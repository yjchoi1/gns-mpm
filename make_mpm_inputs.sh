python3 mpm_input_gen/make_mpm_input.py

MPM_DIR="./mpm/mpm-train22"
mv particles.txt "${MPM_DIR}"
mv entity_sets.json "${MPM_DIR}"
mv mesh.txt "${MPM_DIR}"
