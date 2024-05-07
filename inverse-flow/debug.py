import numpy as np

# Extract one simulation from test.py
path = "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/pipe-npz/"
openfile = "test.npz"
savename = "trj1.npz"

# import data
data = [dict(trj_info.item()) for trj_info in np.load(f"{path}/{openfile}", allow_pickle=True).values()]
trajectory1 = {"trj1": data[1]}

# save data
np.savez_compressed(f"{path}/{savename}", **trajectory1)

# reopen
trj1 = [dict(trj_info.item()) for trj_info in np.load(f"{path}/{savename}", allow_pickle=True).values()]
