python3 mpm_input_gen/mesh-2d.py
python3 mpm_input_gen/particles-2d.py

MPM_DIR="./mpm/mpm-train1"
mv particles.txt "${MPM_DIR}"
mv entity_sets.json "${MPM_DIR}"
mv mesh.txt "${MPM_DIR}"
