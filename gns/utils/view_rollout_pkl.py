import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

# inputs
dir_rollout = "../gns-data/rollouts/"
dataset = "sand-2d-r070"
rollout_name = "rollout_0.pkl"

rollout_path = os.path.join(dir_rollout, dataset, rollout_name)
with open(rollout_path, "rb") as file:
    rollout_data = pickle.load(file)

xs_real = rollout_data['ground_truth_rollout'][:, :, 0]
ys_real = rollout_data['ground_truth_rollout'][:, :, 1]
escaped_particles = np.count_nonzero((0 > xs_real[-1]) or (0 > ys_real[-1]))
xs_pred = rollout_data['predicted_rollout'][:, :, 0]
ys_pred = rollout_data['predicted_rollout'][:, :, 1]

print(f"xmax_real: {np.amax(xs_real)}, xmin_real: {np.amin(xs_real)}")
print(f"ymax_real: {np.amax(ys_real)}, ymin_real: {np.amin(ys_real)}")
print(f"x_real out of boundary: {np.count_nonzero((0 > xs_real[-1]))}")
print(f"y_real out of boundary: {np.count_nonzero((0 > ys_real[-1]))}")

print(f"xmax_pred: {np.amax(xs_pred)}, xmin_pred: {np.amin(xs_pred)}")
print(f"ymax_pred: {np.amax(ys_pred)}, ymin_pred: {np.amin(ys_pred)}")
print(f"x_pred out of boundary: {np.count_nonzero((0 > xs_pred[-1]))}")
print(f"y_pred out of boundary: {np.count_nonzero((0 > ys_pred[-1]))}")
