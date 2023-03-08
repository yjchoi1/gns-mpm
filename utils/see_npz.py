import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/sand2d_frictions0/sand2d_frictions0.npz'
data = dict(np.load(npz_loc, allow_pickle=True))

infos = []
for sim, info in data.items():
    infos.append(info[0])

