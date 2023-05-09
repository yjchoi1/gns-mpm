import json
import pickle
import os
import shutil
import subprocess
import torch
import numpy as np
import sys
import time

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from forward import forward_rollout
from gns import reading_utils
from gns import data_loader
from gns import train


simulation_name = "autograd_short_phi42_a05"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}/"
guess = 32.24109478779155

# inputs for forward rollout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-r015/"
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-r015/"
model_file = "model-5000000.pt"


metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# get ground truth particle position at the last timestep
mpm_trajectory = dict(np.load(f"{path}/{simulation_name}.npz", allow_pickle=True))
for sim, info in mpm_trajectory.items():  # TODO: has only one trajectory. No need to iterate
    target_final_position = info[0][-1]  # info[0] contains positions=(timesteps, nparticles, dims)
    target_final_runout = np.max(target_final_position[:, 0])

# Load data containing MPM initial conditions (six initial positions, particle type, material feature)
dinit = data_loader.TrajectoriesDataset(path=f"{path}/mpm_phi{guess}/mpm_phi{guess}.npz")
with torch.no_grad():
    for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
        # Obtain features
        if len(features) < 3:
            raise NotImplementedError("Data should include material feature")
        positions = features[0].to(device).requires_grad_(False)
        particle_type = features[1].to(device).requires_grad_(False)
        phi_torch = features[2][0].requires_grad_(False)
        n_particles_per_example = torch.tensor(
            [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

        # Compute runout loss
        start_rollout = time.time()
        print(f"Compute rollout at phi: {guess}")
        rollout_data, loss = forward_rollout(
            simulator, positions, particle_type, phi_torch,
            n_particles_per_example, metadata["sequence_length"], target_final_runout,
            device
        )
        end_rollout = time.time()
        rollout_time = end_rollout - start_rollout
        print(f"Rollout for phi {guess} took {rollout_time}s")


        # Save rollout in testing
        rollout_data['metadata'] = metadata
        rollout_data['loss'] = loss.cpu().numpy()
        filename = f'{path}/mpm_phi{guess}/mpm_phi{guess}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(rollout_data, f)
