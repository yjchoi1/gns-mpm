import os
import sys
import numpy as np
import json
import pandas as pd
import pickle
import subprocess
from utils.convert_hd5_to_npz import convert_hd5_to_npz
from render import render_gns_to_mpm


path = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/sand2d_accel_mpm0/"
uuid = "results/sand2d_inverse_eval/"
rollout_path = "results/true_data/rollout_eval7_0_step6300000.pkl"
sample_h5_path = "results/true_data/particles250000.h5"
overwritten_csv = "results/sand2d_inverse_eval/particles250000.csv"
resumed_mpm_npz = "resumed_mpm.npz"
out_step = 99
dt = 0.0025

# get GNS prediction at the specified step
rollout = pickle.load(open(f"{path}{rollout_path}", "rb"))
gns_positions = np.concatenate((rollout["initial_positions"], rollout["predicted_rollout"]))
mpm_positions = np.concatenate((rollout["initial_positions"], rollout["ground_truth_rollout"]))

# specify GNS rollout timestep to pass to MPM
out_position = gns_positions[out_step]
out_vel = (gns_positions[out_step] - gns_positions[out_step-1]) / dt

# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(f'{path}/{sample_h5_path}') as store:
    print(store.keys())
data = pd.read_hdf(f'{path}/{sample_h5_path}', "/")

# pass GNS prediction to MPM h5 file
data["coord_x"] = out_position[:, 0]
data["coord_y"] = out_position[:, 1]
data["velocity_x"] = out_vel[:, 0]
data["velocity_y"] = out_vel[:, 1]
data["stress_xx"] = np.zeros(out_position[:, 0].shape)
data["stress_yy"] = np.zeros(out_position[:, 0].shape)
data["stress_zz"] = np.zeros(out_position[:, 0].shape)
data["tau_xy"] = np.zeros(out_position[:, 0].shape)
data["tau_yz"] = np.zeros(out_position[:, 0].shape)
data["tau_xz"] = np.zeros(out_position[:, 0].shape)
data["strain_xx"] = np.zeros(out_position[:, 0].shape)
data["strain_yy"] = np.zeros(out_position[:, 0].shape)
data["strain_zz"] = np.zeros(out_position[:, 0].shape)
data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

# save it as csv, and convert it to h5
data.to_csv(f"{path}/{overwritten_csv}")
shell_command = f"module load intel hdf5;" \
                f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {path}/{overwritten_csv}"
try:
    subprocess.run(shell_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the shell script: {e}")

# after running MPM by resuming from GNS rollout, get the result
convert_hd5_to_npz(path=path,
                   uuid=uuid,
                   ndim=2,
                   output=f"{path}/{resumed_mpm_npz}",
                   material_feature=True,
                   dt=1.0)
resumed_data = [item for _, item in np.load(f"{path}/{resumed_mpm_npz}", allow_pickle=True).items()]
resumed_positions = resumed_data[0][0]

# concat resumed data to gns rollout
gns_to_mpm_positions = np.concatenate((gns_positions[:out_step], resumed_positions))
rollout_steps = len(gns_to_mpm_positions)
position_results = {
    "mpm_rollout": mpm_positions[:rollout_steps],
    "gns_to_mpm_rollout": gns_to_mpm_positions
}

# render
render_gns_to_mpm(data=position_results,
                  boundaries=[[0, 1], [0, 1]],
                  output_name=f"{path}/gns_to_mpm.gif")
