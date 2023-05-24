import pickle
import numpy as np

file_loc = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand3d-largesets-r041/"
file_name = "rollout_column_collapse12_0_step8690000"
with open(f"{file_loc}{file_name}.pkl", "rb") as file:
    rollout_data = pickle.load(file)

max_x_mpm = rollout_data["ground_truth_rollout"][-1, :, 0].max()
max_x_gns = rollout_data["predicted_rollout"][-1, :, 0].max()

L0 = 0.475
runout_mpm = (max_x_mpm-L0)/L0
runout_gns = (max_x_gns-L0)/L0