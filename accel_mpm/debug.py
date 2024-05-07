import pandas as pd
import numpy as np
import subprocess

path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/accel_mpm/sand2d_inverse_eval7/"
sample_h5_path = f"{path}/results/particles00000.h5"
save_cvs_path = f"{path}/results/particles{2500*105}.csv"
out_step = -1
dt = 0.0025

# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(sample_h5_path) as store:
    print(store.keys())
data = pd.read_hdf(sample_h5_path, "/")
