import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand3d-largesets-r041/train.npz'
initial_timesteps = 6
_data = [item for _, item in np.load(npz_loc, allow_pickle=True).items()]
data = dict(np.load(npz_loc, allow_pickle=True))

trajectories = {}
for i, (sim, info) in enumerate(data.items()):
    if i == 4:
        print(info[0].shape)
        print(info[1].shape)
        # print(info[2].shape)
        max_runout = info[0][-1][:, ].max()
        print(max_runout)
        trj = info[0]
    else:
        pass
    # trajectories[sim] = (info[0][:initial_timesteps], info[1], info[2])
# np.savez_compressed("initial_conditions.npz", **trajectories)

timesteps_to_plot = [0, 50, 130, 359]
fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '3d'}, figsize=(9, 2.5))
for i, ax in enumerate(axs):
    ax.scatter(trj[timesteps_to_plot[i]][:, 0],
               trj[timesteps_to_plot[i]][:, 1],
               trj[timesteps_to_plot[i]][:, 2], s=1.0)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
# plt.tight_layout()
plt.show()
