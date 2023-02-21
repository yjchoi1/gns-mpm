import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

npz_loc = '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand-small-r300-400step_serial/train.npz'
data = dict(np.load(npz_loc, allow_pickle=True))