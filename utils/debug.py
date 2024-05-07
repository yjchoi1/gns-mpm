import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation

npz_locs = [
    '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/short_phi21/sand2d_inverse_eval25.npz',
    '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/short_phi42/sand2d_inverse_eval13.npz',
    '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi21/sand2d_inverse_eval3.npz',
    '/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi42/sand2d_inverse_eval15.npz'
    ]
for npz_loc in npz_locs:
    data = dict(np.load(npz_loc, allow_pickle=True))
    for i, (sim, info) in enumerate(data.items()):
        max_runout = info[0][-1][:, ].max()
        print(f"Max runout: {max_runout}")