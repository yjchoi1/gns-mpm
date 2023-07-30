import os
import sys
import torch
import numpy as np
import time
import json
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import subprocess
from utils.convert_hd5_to_npz import convert_hd5_to_npz
from accel_mpm.render import render_gns_to_mpm
from accel_mpm.convert_npz_to_h5 import make_resume_h5_from_npz


sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import reading_utils
from gns import data_loader
from gns import train
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/inverse/')
from forward import forward_rollout


# input
path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/accel_mpm/sand2d_inverse_eval7/"
uuid = "results/sand2d_inverse_eval/"

# GNS inputs
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/"
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/"
model_file = "model-6300000.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9



#%% [Run MPM for X0]
# start_mpm = time.time()
#
# # Write bash script
# with open(f'{path}/run_mpm.sh', 'w') as rsh:
#     rsh.write(
#         f'''\
#         module reset
#         module load intel
#         module load libfabric
#
#         timeout 10 /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "{path}/"
#         /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input_resume.json -f "{path}/"
#         ''')
#
# # Read bash script
# with open(f'{path}/run_mpm.sh', 'rb') as bashfile:
#     script = bashfile.read()
# # Run mpm
# with open(f"{path}/mpm_out_x0.txt", 'w') as outfile:
#     rc = subprocess.call(script, shell=True, stdout=outfile)
#
# end_mpm = time.time()
# mpm_time = end_mpm - start_mpm
# print(f"MPM for X0 took {mpm_time}s")

#%% [Convert npz from h5]
convert_hd5_to_npz(
    path=path,
    uuid=uuid,
    ndim=2,
    output=f"{path}/x0.npz",
    material_feature=True,
    dt=1.0)

#%% [Run GNS for 100 steps (X0 -> X100, p0 -> p105)]
nforward_steps = 100
x0_path = f"{path}/x0.npz"

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get simulator input
dinit = data_loader.TrajectoriesDataset(path=x0_path)
for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(False)
    particle_type = features[1].to(device).requires_grad_(False)
    phi_torch = features[2][0].requires_grad_(False)
    n_particles_per_example = torch.tensor(
        [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)
#%%
# Rollout
with torch.no_grad():
    rollout_data_x0, _ = forward_rollout(
              simulator, positions[:, -6:, :], particle_type, phi_torch,
              n_particles_per_example, nforward_steps, None,
              device)

#%% [Convert npz to h5 to make particles<2500*105=262500>.h5]
from matplotlib import pyplot as plt
sample_h5_path = f"{path}/results/particles002500.h5"
save_cvs_path = f"{path}/results/particles{2500*105}.csv"
out_step = -1
dt = 0.0025

# get position and vel data to pass to h5 for resuming
gns_positions = np.concatenate((rollout_data_x0["initial_positions"], rollout_data_x0["predicted_rollout"]))
out_position = gns_positions[out_step]
out_vel = (gns_positions[out_step] - gns_positions[out_step - 1]) / dt

# # plot
# timesteps_to_plot = [10, 50, -1]
# # fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '2d'}, figsize=(9, 2.5))
# for t in timesteps_to_plot:
#     fig, axs = plt.subplots(1, 1)
#     axs.scatter(gns_positions[t][:, 0],
#                 gns_positions[t][:, 1], s=1.0)
#     # trj[timesteps_to_plot[i]][:, 2], s=1.0)
#     axs.set_xlim([0, 1])
#     axs.set_ylim([0, 1])
#     plt.show()
#%%
# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(sample_h5_path) as store:
    print(store.keys())
sample_h5_data = pd.read_hdf(sample_h5_path, "/")

# pass GNS prediction to MPM h5 file
sample_h5_data["coord_x"] = out_position[:, 0]
sample_h5_data["coord_y"] = out_position[:, 1]
sample_h5_data["velocity_x"] = out_vel[:, 0]
sample_h5_data["velocity_y"] = out_vel[:, 1]
sample_h5_data["stress_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

# save it as csv, and convert it to h5
sample_h5_data.to_csv(save_cvs_path)
shell_command = f"module load intel hdf5;" \
                f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {save_cvs_path}"
try:
    subprocess.run(shell_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the shell script: {e}")

#%% [Resume MPM and run for 6 steps (2500*105=262500 -> 2500*110=275000)]

#%% [Convert npz from h5]
convert_hd5_to_npz(
    path=path,
    uuid=uuid,
    ndim=2,
    output=f"{path}/x105.npz",
    material_feature=True,
    dt=1.0)

#%% [Run GNS for 100 steps (X105 -> X205, p110 -> p210)]
nforward_steps = 100
x0_path = f"{path}/x105.npz"

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get simulator input
dinit = data_loader.TrajectoriesDataset(path=x0_path)
for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(False)
    particle_type = features[1].to(device).requires_grad_(False)
    phi_torch = features[2][0].requires_grad_(False)
    n_particles_per_example = torch.tensor(
        [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

#%%
# Rollout
with torch.no_grad():
    rollout_data_x105, _ = forward_rollout(
              simulator, positions[:, -6:, :], particle_type, phi_torch,
              n_particles_per_example, nforward_steps, None,
              device)

#%% [Convert npz to h5 to make particles<2500*210=525000>.h5]
sample_h5_path = f"{path}/results/particles002500.h5"
save_cvs_path = f"{path}/results/particles{2500*210}.csv"
out_step = -1
dt = 0.0025

# get position and vel data to pass to h5 for resuming
gns_positions = np.concatenate((rollout_data_x105["initial_positions"], rollout_data_x105["predicted_rollout"]))
out_position = gns_positions[out_step]
out_vel = (gns_positions[out_step] - gns_positions[out_step - 1]) / dt

# # plot
# timesteps_to_plot = [-20, -10, -5, -3, -2, -1]
# # fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '2d'}, figsize=(9, 2.5))
# for t in timesteps_to_plot:
#     fig, axs = plt.subplots(1, 1)
#     axs.scatter(gns_positions[t][:, 0],
#                 gns_positions[t][:, 1], s=1.0)
#     # trj[timesteps_to_plot[i]][:, 2], s=1.0)
#     axs.set_xlim([0, 1])
#     axs.set_ylim([0, 1])
#     plt.show()
#%%
# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(sample_h5_path) as store:
    print(store.keys())
sample_h5_data = pd.read_hdf(sample_h5_path, "/")

# pass GNS prediction to MPM h5 file
sample_h5_data["coord_x"] = out_position[:, 0]
sample_h5_data["coord_y"] = out_position[:, 1]
sample_h5_data["velocity_x"] = out_vel[:, 0]
sample_h5_data["velocity_y"] = out_vel[:, 1]
sample_h5_data["stress_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

# save it as csv, and convert it to h5
sample_h5_data.to_csv(save_cvs_path)
shell_command = f"module load intel hdf5;" \
                f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {save_cvs_path}"
try:
    subprocess.run(shell_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the shell script: {e}")

#%% [Resume MPM and run for 6 steps (2500*210=525000 -> 2500*215=527500)]

###################################
#%% [Convert npz from h5]
convert_hd5_to_npz(
    path=path,
    uuid=uuid,
    ndim=2,
    output=f"{path}/x210.npz",
    material_feature=True,
    dt=1.0)

#%% [Run GNS for 100 steps (X210 -> X210, p215 -> p315)]
nforward_steps = 100
x0_path = f"{path}/x210.npz"

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get simulator input
dinit = data_loader.TrajectoriesDataset(path=x0_path)
for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(False)
    particle_type = features[1].to(device).requires_grad_(False)
    phi_torch = features[2][0].requires_grad_(False)
    n_particles_per_example = torch.tensor(
        [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

#%%
# Rollout
with torch.no_grad():
    rollout_data_x210, _ = forward_rollout(
              simulator, positions[:, -6:, :], particle_type, phi_torch,
              n_particles_per_example, nforward_steps, None,
              device)

#%% [Convert npz to h5 to make particles<2500*315=787500>.h5
sample_h5_path = f"{path}/results/particles002500.h5"
save_cvs_path = f"{path}/results/particles{2500*315}.csv"
out_step = -1
dt = 0.0025

# get position and vel data to pass to h5 for resuming
gns_positions = np.concatenate((rollout_data_x105["initial_positions"], rollout_data_x105["predicted_rollout"]))
out_position = gns_positions[out_step]
out_vel = (gns_positions[out_step] - gns_positions[out_step - 1]) / dt

# # plot
# timesteps_to_plot = [-20, -10, -5, -3, -2, -1]
# # fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '2d'}, figsize=(9, 2.5))
# for t in timesteps_to_plot:
#     fig, axs = plt.subplots(1, 1)
#     axs.scatter(gns_positions[t][:, 0],
#                 gns_positions[t][:, 1], s=1.0)
#     # trj[timesteps_to_plot[i]][:, 2], s=1.0)
#     axs.set_xlim([0, 1])
#     axs.set_ylim([0, 1])
#     plt.show()

#%%
# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(sample_h5_path) as store:
    print(store.keys())
sample_h5_data = pd.read_hdf(sample_h5_path, "/")

# pass GNS prediction to MPM h5 file
sample_h5_data["coord_x"] = out_position[:, 0]
sample_h5_data["coord_y"] = out_position[:, 1]
sample_h5_data["velocity_x"] = out_vel[:, 0]
sample_h5_data["velocity_y"] = out_vel[:, 1]
sample_h5_data["stress_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

# save it as csv, and convert it to h5
sample_h5_data.to_csv(save_cvs_path)
shell_command = f"module load intel hdf5;" \
                f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {save_cvs_path}"
try:
    subprocess.run(shell_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the shell script: {e}")
#%% [Resume MPM and run for 6 steps (2500*315=787500 -> 2500*320=800000)]

###################################

#%% [Convert npz from h5]
convert_hd5_to_npz(
    path=path,
    uuid=uuid,
    ndim=2,
    output=f"{path}/x315.npz",
    material_feature=True,
    dt=1.0)

#%% [Run GNS for 100 steps (X315 -> X369, p320 -> p374)]
nforward_steps = 54
x0_path = f"{path}/x315.npz"

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get simulator input
dinit = data_loader.TrajectoriesDataset(path=x0_path)
for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(False)
    particle_type = features[1].to(device).requires_grad_(False)
    phi_torch = features[2][0].requires_grad_(False)
    n_particles_per_example = torch.tensor(
        [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

#%%
# Rollout
with torch.no_grad():
    rollout_data_x315, _ = forward_rollout(
              simulator, positions[:, -6:, :], particle_type, phi_torch,
              n_particles_per_example, nforward_steps, None,
              device)

#%% [Convert npz to h5 to make particles<2500*374=935000>.h5
sample_h5_path = f"{path}/results/particles002500.h5"
save_cvs_path = f"{path}/results/particles{2500*374}.csv"
out_step = -1
dt = 0.0025

# get position and vel data to pass to h5 for resuming
gns_positions = np.concatenate((rollout_data_x105["initial_positions"], rollout_data_x105["predicted_rollout"]))
out_position = gns_positions[out_step]
out_vel = (gns_positions[out_step] - gns_positions[out_step - 1]) / dt

#%%
# get arbitrary sample mpm h5 file to overwrite GNS rollout
with pd.HDFStore(sample_h5_path) as store:
    print(store.keys())
sample_h5_data = pd.read_hdf(sample_h5_path, "/")

# pass GNS prediction to MPM h5 file
sample_h5_data["coord_x"] = out_position[:, 0]
sample_h5_data["coord_y"] = out_position[:, 1]
sample_h5_data["velocity_x"] = out_vel[:, 0]
sample_h5_data["velocity_y"] = out_vel[:, 1]
sample_h5_data["stress_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["stress_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["tau_xz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_xx"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_yy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["strain_zz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
sample_h5_data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

# save it as csv, and convert it to h5
sample_h5_data.to_csv(save_cvs_path)
shell_command = f"module load intel hdf5;" \
                f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {save_cvs_path}"
try:
    subprocess.run(shell_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error while running the shell script: {e}")

#%% [Resume MPM and run for 6 steps (2500*374=935000 -> 2500*379=947500)]

###################

#%% [Convert npz from h5]
convert_hd5_to_npz(
    path=path,
    uuid=uuid,
    ndim=2,
    output=f"{path}/x374.npz",
    material_feature=True,
    dt=1.0)

#%% [Pure gns rollout from x0]
nforward_steps = 374
x0_path = f"{path}/x0.npz"

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get simulator input
dinit = data_loader.TrajectoriesDataset(path=x0_path)
for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(False)
    particle_type = features[1].to(device).requires_grad_(False)
    phi_torch = features[2][0].requires_grad_(False)
    n_particles_per_example = torch.tensor(
        [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

with torch.no_grad():
    rollout_data_pure_gns, _ = forward_rollout(
              simulator, positions[:, -6:, :], particle_type, phi_torch,
              n_particles_per_example, nforward_steps, None,
              device)


#%% [Process the result and save]
# mpm
mpm_positions_data_path = f"{path}/mpm_x0to379.npz"
mpm_positions_data = [item for _, item in np.load(mpm_positions_data_path, allow_pickle=True).items()]
mpm_positions = mpm_positions_data[0][0]

# gns
gns_positions = np.concatenate((rollout_data_pure_gns["initial_positions"], rollout_data_pure_gns["predicted_rollout"]))

# gns+mpm
p0to105 = np.concatenate((rollout_data_x0["initial_positions"], rollout_data_x0["predicted_rollout"]))
p105to210 = np.concatenate((rollout_data_x105["initial_positions"], rollout_data_x105["predicted_rollout"]))
p210to315 = np.concatenate((rollout_data_x210["initial_positions"], rollout_data_x210["predicted_rollout"]))
p315to374 = np.concatenate((rollout_data_x315["initial_positions"], rollout_data_x315["predicted_rollout"]))
data374to379 = [item for _, item in np.load(f"{path}/x374.npz", allow_pickle=True).items()]
p374to379 = data374to379[0][0][-6:]
gns_to_mpm_positions = np.concatenate((p0to105[:-1], p105to210[:-1], p210to315[:-1], p315to374[:-1], p374to379))

# save
position_results = {
    "mpm_rollout": mpm_positions,
    "gns_rollout": gns_positions,
    "gns_to_mpm_rollout": gns_to_mpm_positions
}
with open(f'{path}/position_results.pkl', 'wb') as handle:
    pickle.dump(position_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% [Load result for analysis and visualize]
import pickle

with open(f'{path}/position_results.pkl', 'rb') as f:
    position_results = pickle.load(f)

# render
render_gns_to_mpm(data=position_results,
                  boundaries=[[0, 1], [0, 1]],
                  output_name=f"{path}/gns_to_mpm.gif")

#%%
# rollout analysis for "MPM" vs "GNS+MPM"
analysis_data = {
    "initial_positions": position_results["mpm_rollout"][:6, :, :],
    "predicted_rollout": position_results["gns_to_mpm_rollout"][6:, :, :],
    "ground_truth_rollout": position_results["mpm_rollout"][6:, :, :],
    "particle_types": np.full(position_results["mpm_rollout"].shape[1], 6),
    "metadata": metadata
}
with open(f'{path}/position_results_processed.pkl', 'wb') as handle:
    pickle.dump(analysis_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Error evolution

# get an initial position (nparticles, 2) and expand it to (ntimesteps, nparticles, 2)
# For unknown reason current x0 from MPM is different from the previous file though it should be the same,-
# so, use different initial position for the current case and the case for MPM

ntimesteps = len(position_results["mpm_rollout"])
initial_position = position_results["mpm_rollout"][0]
initial_position_expanded = np.tile(initial_position[None, :, :], (ntimesteps, 1, 1))

# compute the displacements
disp_mpm = np.linalg.norm(
    initial_position_expanded - position_results["mpm_rollout"], axis=-1)
disp_gns_to_mpm = np.linalg.norm(
    initial_position_expanded - position_results["gns_to_mpm_rollout"], axis=-1)
disp_gns_only = np.linalg.norm(
    initial_position_expanded - position_results["gns_rollout"], axis=-1)


# compute relative error
error_gns_to_mpm = np.nanmean(np.abs((disp_gns_to_mpm - disp_mpm)/disp_mpm), axis=1)*100
error_gns_only = np.nanmean(np.abs((disp_gns_only - disp_mpm)/disp_mpm), axis=1)*100

# # compute absolute position error
# error_gns_to_mpm = np.linalg.norm(
#     position_results["mpm_rollout"] - position_results["gns_to_mpm_rollout"], axis=-1
# ).mean(axis=1)
# error_gns_only = np.linalg.norm(
#     position_results["mpm_rollout"] - position_results["gns_rollout"], axis=-1
# ).mean(axis=1)

# plot
titles = ["GNS+MPM", "GNS-only"]
timesteps = range(ntimesteps)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3))
axs[0].plot(timesteps, error_gns_to_mpm)
axs[1].plot(timesteps, error_gns_only)
for i, ax in enumerate(axs):
    if i == 0:
        ax.axvline(x=5, c="black", alpha=0.5)
        ax.axvline(x=105, c="black", alpha=0.5)
        ax.axvline(x=110, c="black", alpha=0.5)
        ax.axvline(x=210, c="black", alpha=0.5)
        ax.axvline(x=215, c="black", alpha=0.5)
        ax.axvline(x=315, c="black", alpha=0.5)
        ax.axvline(x=320, c="black", alpha=0.5)
        ax.axvline(x=374, c="black", alpha=0.5)
        ax.axvline(x=379, c="black", alpha=0.5)
    ax.set_xlim([0, 380])
    ax.set_ylim([0, 500])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Displacement error (%)")
    ax.set_title(titles[i])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.savefig(f"{path}/error.png")
plt.show()


