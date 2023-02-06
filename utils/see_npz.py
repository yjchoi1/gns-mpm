import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/droplet2/merged_train2d.npz'
data = dict(np.load(npz_loc, allow_pickle=True))