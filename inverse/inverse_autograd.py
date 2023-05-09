import argparse

from absl import app
import torch
import os
import sys
import numpy as np
import json
import time
import pickle

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from forward import forward_rollout, forward_rollout_autograd
from free_gpu_cache import free_gpu_cache
from gns import reading_utils
from gns import data_loader
from gns import train
from convert_hd5_to_npz import convert_hd5_to_npz
from animation_from_pkl import animation_from_pkl
from run_mpm import run_mpm
from memory_profiler import profile


import argparse



def autograd_inverse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=None,
                        help="initial phi guess")
    parser.add_argument("--phi", type=float, default=None,
                        help="initial phi guess")
    args = parser.parse_args()

    if args.epoch == 0:
        resume = False
        phi = args.phi  # initial guess of phi
    else:
        resume = True
    lr = 100000000  # learning rate
    simulation_name = "sand2d_frictions_autograd_gpu"
    path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}/"
    mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
    analysis_dt = 1e-06
    output_steps = 2500
    analysis_nsteps = 2500*5 + 1  # only run to get 6 initial positions to make X_0 in GNS

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
    nforward_steps = 5
    target_step = nforward_steps

    ### End of inputs ###

    # Make a record file to save mpm input files, current phi, loss, gradient during gradient descent
    record_file = os.path.join(path, "record.pkl")
    if not resume:
        records = []
        with open(record_file, 'wb') as f:
            pickle.dump(records, f)
        f.close()
    else:
        records = pickle.load(open(record_file, "rb"))
        phi = records[-1]["updated_phi"]

    # Start forward computations and gradient descent
    epoch = args.epoch
    print(f"Epoch {epoch}---------------------------------------------------")
    start_epoch = time.time()

    # start recording inverse analysis information for current phi
    record = {}
    record[f"epoch"] = epoch
    record["current_phi"] = phi

    # RUM MPM FOR CURRENT PHI GUESS
    start_mpm = time.time()
    record = run_mpm(path, mpm_input,
                     phi, analysis_dt, analysis_nsteps, output_steps,
                     record=record)
    print(f"Running MPM for {phi} at epoch {epoch}...")
    end_mpm = time.time()
    mpm_time = end_mpm - start_mpm
    print(f"MPM for phi {phi} took {mpm_time}s")
    record[f"mpm_time"] = mpm_time

    # make `.npz` to prepare initial state X_1 for rollout
    sim_name = f"mpm_phi{phi}"
    uuid = "/results/sand2d_frictions_test"
    convert_hd5_to_npz(path=path + sim_name,
                       uuid=uuid,
                       ndim=ndim,
                       output=f"{path}{sim_name}/{sim_name}.npz",
                       material_feature=material_feature,
                       dt=dt
                       )

    # %% ROLLOUT FOR EACH PHI GUESS
    metadata = reading_utils.read_metadata(path)
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
        target_final_position = info[0][target_step]  # info[0] contains positions=(timesteps, nparticles, dims)

    # Load data containing MPM initial conditions (six initial positions, particle type, material feature)
    dinit = data_loader.TrajectoriesDataset(path=f"{path}/mpm_phi{phi}/mpm_phi{phi}.npz")
    for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need for loop
        # Obtain features
        if len(features) < 3:
            raise NotImplementedError("Data should include material feature")
        initial_positions = features[0].to(device).detach().requires_grad_(False)
        particle_type = features[1].to(device).detach().requires_grad_(False)
        material_property = features[2].to(device).detach().requires_grad_(False)
        n_particles_per_example = torch.tensor(
            [int(features[3])], dtype=torch.int32).to(device).detach().requires_grad_(False)

    phi_guess = material_property[0].clone().to(device).requires_grad_(True)
    start_rollout = time.time()
    print(f"Compute rollout at phi: {phi}")
    loss, predicted_final_positions = forward_rollout_autograd(
        simulator, initial_positions, particle_type, phi_guess,
        n_particles_per_example, nforward_steps, target_final_position,
        device)

    end_rollout = time.time()
    rollout_time = end_rollout - start_rollout
    print(f"Rollout for phi {phi} took {rollout_time}s")
    record[f"rollout_time"] = rollout_time

    print(f"Loss: {loss}")
    loss.to(device).requires_grad_(True)
    loss.backward(retain_graph=True, inputs=[phi_guess])
    loss.detach()

    # Access gradients of input material_type
    grads = phi_guess.grad.cpu().detach().numpy()
    phi = phi - lr * grads
    print(f"Gradient of loss w.r.t phi is: {grads}")

    end_epoch = time.time()
    print(f"Epoch {epoch} took {end_epoch - start_epoch}s")

    # record rollout information
    record[f"loss"] = loss.cpu().detach().numpy()
    record[f"rollout"] = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_final_positions": predicted_final_positions,
        "particle_types": particle_type.cpu().numpy(),
        'material_property': material_property.cpu().numpy(),
        "target_final_positions": target_final_position
    }
    record["updated_phi"] = phi
    records.append(record)

    # save current record file
    with open(record_file, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    # # free memory
    # free_gpu_cache()
    # simulator = simulator.to("cpu")
    # del simulator



if __name__ == '__main__':

    autograd_inverse()

#
#
# guess = torch.tensor(float(phi), requires_grad=True)
# # %% ROLLOUT FOR EACH PHI GUESS
# # Load simulator
# metadata = reading_utils.read_metadata(path)
# simulator = train._get_simulator(metadata, noise_std, noise_std, device)
# if os.path.exists(model_path + model_file):
#     simulator.load(model_path + model_file)
# else:
#     raise Exception(f"Model does not exist at {model_path + model_file}")
# simulator.to(device)
# simulator.eval()
#
# # get ground truth particle position at the last timestep
# mpm_trajectory = dict(np.load(f"{path}/{simulation_name}.npz", allow_pickle=True))
# for sim, info in mpm_trajectory.items():  # has only one trajectory. No need to iterate
#     target_final_position = info[0][-1]  # info[0] contains positions=(timesteps, nparticles, dims)
#
# # Load data containing MPM initial conditions (six initial positions, particle type, material feature)
# dinit = data_loader.TrajectoriesDataset(path=f"{path}/mpm_phi{guess}/mpm_phi{guess}.npz")
#
# for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need for loop
#     # Obtain features
#     if len(features) < 3:
#         raise NotImplementedError("Data should include material feature")
#     positions = features[0].to(device).requires_grad_(False)
#     particle_type = features[1].to(device).requires_grad_(False)
#     phi = features[2][0].clone().to(device).requires_grad_(True)
#     n_particles_per_example = torch.tensor(
#         [int(features[3])], dtype=torch.int32).to(device)
#
#     # Compute runout loss
#     start_rollout = time.time()
#     print(f"Compute rollout at phi: {guess}")
#     loss = forward_rollout_autograd(
#         simulator, positions, particle_type, phi,
#         n_particles_per_example, nforward_steps, target_final_position,
#         device
#     )
#
#     print(f"Loss: {loss}")
#     loss.to(device).requires_grad_(True)
#     loss.backward(retain_graph=True, inputs=[guess])
#
#     # Access gradients of input material_type
#     grads = guess.grad
#
#     # Print the gradients
#     print(f"Gradient of loss w.r.t phi is: {grads}")
#
