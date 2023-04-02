import torch
import os
import sys
import numpy as np
import json
import time

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from forward import forward_rollout, forward_rollout_autograd
from gns import reading_utils
from gns import data_loader
from gns import train

simulation_name = "sand2d_frictions_autograd_test1"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}"
phi = 45  # initial guess of phi
mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
analysis_nsteps = 12501  # only run to get 6 initial positions to make X_0 in GNS
analysis_dt = 1e-06
output_steps = 2500

# inputs for make `.npz` containing initial 6 steps. This npz file corresponds to X_0
material_feature = True
ndim = 2
dt = 1.0

# inputs for forward rollout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-r015/"
model_file = "model-4300000.pt"

nsteps = 380


guess = torch.tensor(float(phi), requires_grad=True)
# %% ROLLOUT FOR EACH PHI GUESS
# Load simulator
metadata = reading_utils.read_metadata(path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()  # ??

# get ground truth particle position at the last timestep
mpm_trajectory = dict(np.load(f"{path}/{simulation_name}.npz", allow_pickle=True))
for sim, info in mpm_trajectory.items():  # has only one trajectory. No need to iterate
    target_final_position = info[0][-1]  # info[0] contains positions=(timesteps, nparticles, dims)

# Load data containing MPM initial conditions (six initial positions, particle type, material feature)
dinit = data_loader.TrajectoriesDataset(path=f"{path}/mpm_phi{guess}/mpm_phi{guess}.npz")

for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need for loop
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device).requires_grad_(True)
    particle_type = features[1].to(device).requires_grad_(True)
    phi_torch = features[2][0].requires_grad_(True)
    n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device).requires_grad_(
        True)

    # Compute runout loss
    start_rollout = time.time()
    print(f"Compute rollout at phi: {guess}")
    with torch.no_grad():
        loss = forward_rollout_autograd(
            simulator, positions, particle_type, phi_torch,
            n_particles_per_example, metadata["sequence_length"], target_final_position,
            device
        )

    print(f"Loss: {loss}")
    loss.to(device).requires_grad_(True)
    loss.backward(retain_graph=True, inputs=[guess])

    # Access gradients of input material_type
    grads = guess.grad

    # Print the gradients
    print(f"Gradient of loss w.r.t phi is: {grads}")

