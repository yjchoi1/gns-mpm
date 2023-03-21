import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/sand2d_frictions338/initial_conditions.npz'
initial_timesteps = 6
data = dict(np.load(npz_loc, allow_pickle=True))

trajectories = {}
for i, (sim, info) in enumerate(data.items()):
    print(len(info[0]))
    print(len(info[1]))
    print(len(info[2]))
    # trajectories[sim] = (info[0][:initial_timesteps], info[1], info[2])
# np.savez_compressed("initial_conditions.npz", **trajectories)

