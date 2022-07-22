# GNS-MPM Implementation with 2D Sand column Collapse

# Dataset
* Dataset should be downloaded with the make `gns-data`.
	* Link:
* `mpm` folder should be exist.

# Implementation
1. `source make_mpm_input.sh`: make MPM inputs (e.g., mesh.txt, particles.txt, entity_sets.json).
2. `source run_mpm.sh`: run MPM analysis.
3. `source make_npz.sh`: make `.npz` file for GNS inputs.
4. `source rollout-2d.sh`: rollout
