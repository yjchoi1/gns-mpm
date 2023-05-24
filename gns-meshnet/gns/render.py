import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from tqdm import tqdm
import glob
import os

rollout_path = "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/rollouts/pipe-npz/"
result_filename = "rollout_0.pkl"
result_path = f"{rollout_path}/{result_filename}"

with open(result_path, 'rb') as f:
    result = pickle.load(f)

# color
vmin = result["predicted_rollout"][0][:, 0].min()
vmax = result["predicted_rollout"][0][:, 0].max()
triang = tri.Triangulation(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])


skip = 100

for i, (pred_vel, target_vel) in enumerate(zip(result["predicted_rollout"], result["ground_truth_rollout"])):

    if i % skip == 0:
        print(i)
        fig, axes = plt.subplots(2, 1, figsize=(17, 8))

        target_vel_mag = np.linalg.norm(target_vel, axis=-1)
        predicted_vel_mag = np.linalg.norm(pred_vel, axis=-1)

        for ax in axes:
            ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)

        handle1 = axes[0].tripcolor(triang, target_vel_mag, vmax=vmax, vmin=vmin)
        axes[1].tripcolor(triang, predicted_vel_mag, vmax=vmax, vmin=vmin)
        plt.show()
    else:
        pass



