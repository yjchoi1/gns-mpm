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
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/inverse/')
from forward import forward_rollout
from convert_hd5_to_npz import convert_hd5_to_npz
from animation_from_pkl import animation_from_pkl
from run_mpm import run_mpm
from gns import reading_utils
from gns import data_loader
from gns import train


simulation_name = "see_runout_gradient_over_phi_short_step30"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}/"
eval_phi_list = np.linspace(18, 45, 10)

# inputs for forward rollout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/"
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/"
model_file = "model-7020000.pt"
nforward_steps = 30

# mpm input
mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
analysis_nsteps = 12501  # only run to get 6 initial positions to make X_0 in GNS
analysis_dt = 1e-06
output_steps = 2500
uuid_name = "sand2d_inverse_eval"

# inputs for make `.npz` containing initial 6 steps. This npz file corresponds to X_0
material_feature = True
ndim = 2
dt = 1.0

# inputs for render
xbound = [-0.0025, 1.0025]
ybound = [-0.0025, 0.5025]


# load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Make a record file
record_file = os.path.join(path, "record.pkl")
records = []
with open(record_file, 'wb') as f:
    pickle.dump(records, f)
f.close()

# Start forward computations
for phi in eval_phi_list:

    # start recording inverse analysis information for current phi
    record = {}

    # RUN MPM for current phi guess
    _ = run_mpm(path, mpm_input,
                phi, analysis_dt, analysis_nsteps, output_steps,
                record=record)
    print(f"Running MPM for phi: {phi}...")

    # make `.npz` to prepare initial state X_1 for rollout
    sim_name = f"mpm_phi{phi}"
    uuid = f"/results/{uuid_name}"
    convert_hd5_to_npz(path=path + sim_name,
                       uuid=uuid,
                       ndim=ndim,
                       output=f"{path}{sim_name}/{sim_name}.npz",
                       material_feature=material_feature,
                       dt=dt)

    # Load data containing MPM initial conditions (six initial positions, particle type, material feature)
    dinit = data_loader.TrajectoriesDataset(path=f"{path}/mpm_phi{phi}/mpm_phi{phi}.npz")

    # ROLLOUT
    with torch.no_grad():
        # get initial condition
        for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
            # Obtain features
            if len(features) < 3:
                raise NotImplementedError("Data should include material feature")
            positions = features[0].to(device).requires_grad_(False)
            particle_type = features[1].to(device).requires_grad_(False)
            phi_torch = features[2][0].requires_grad_(False)
            n_particles_per_example = torch.tensor(
                [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)

        # rollout
        print(f"Compute rollout at phi: {phi}")
        rollout_data, _ = forward_rollout(
          simulator, positions, particle_type, phi_torch,
          n_particles_per_example, nforward_steps, None,
          device)

    # save predicted rollout
    filename = f'{path}/mpm_phi{phi}/mpm_phi{phi}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(rollout_data, f)
    # get final runout distance
    final_runout = rollout_data["predicted_rollout"][-1][:, 0].max()

    # See how the rollout looks like by animation
    print(f"Rendering at phi: {phi}")
    render_start = time.time()
    animation_from_pkl(path=f'{path}/mpm_phi{phi}/mpm_phi{phi}.pkl',
                       output=f'{path}/mpm_phi{phi}/',
                       xbound=xbound,
                       ybound=ybound)

    # Save results
    record["phi"] = phi
    record["final_runout"] = final_runout

    # append current epoch's record
    records.append(record)

    # save current record file
    with open(record_file, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


