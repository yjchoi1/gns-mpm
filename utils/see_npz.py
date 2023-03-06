import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm-data/mpm/mpm-small-test5-5/mpm-small-test5-5.npz'
data = dict(np.load(npz_loc, allow_pickle=True))

infos = []
for sim, info in data.items():
    infos.append(info[0])

