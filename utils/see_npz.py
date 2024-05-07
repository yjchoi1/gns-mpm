import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-inverse-examples/inverse_barrier/data/ground_truth.npz'
initial_timesteps = 6
_data = [item for _, item in np.load(npz_loc, allow_pickle=True).items()]
data = dict(np.load(npz_loc, allow_pickle=True))

trajectories = {}
for i, (sim, info) in enumerate(data.items()):
    print(info[0].shape)
    print(info[1].shape)
    # print(info[2].shape)
    max_runout = info[0][-1][:, ].max()
    print(max_runout)
    trj = info[0]
    # trajectories[sim] = (info[0][:initial_timesteps], info[1], info[2])
# np.savez_compressed("initial_conditions.npz", **trajectories)

timesteps_to_plot = [0, 400, -1]
# fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '2d'}, figsize=(9, 2.5))
for t in timesteps_to_plot:
    fig, axs = plt.subplots(1, 1)
    axs.scatter(trj[t][:, 0],
                trj[t][:, 1], s=1.0)
    # trj[timesteps_to_plot[i]][:, 2], s=1.0)
    axs.set_xlim([-250, 250])
    axs.set_ylim([0, 150])
    plt.show()

a = 1
# 3d
# for i, ax in enumerate(axs):
#     ax.scatter(trj[timesteps_to_plot[i]][:, 0],
#                trj[timesteps_to_plot[i]][:, 1], s=1.0)
#                # trj[timesteps_to_plot[i]][:, 2], s=1.0)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     # ax.set_zlim([0, 1])
#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("y (m)")
#     # ax.set_zlabel("z (m)")
# # plt.tight_layout()
# plt.show()
