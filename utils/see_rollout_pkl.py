import pickle
import numpy as np

file_loc = "./gns-data/rollouts/sand-small-r300-400step_parallel/"
file_name = "rollout_test4-2_0_step5000000"
with open(f"{file_loc}{file_name}.pkl", "rb") as file:
    rollout_data = pickle.load(file)