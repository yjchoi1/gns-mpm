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
from animation_from_pkl import plot_final_position


import argparse



def autograd_inverse():
    resume = False
    nepochs = 20
    phi = 30.0
    lr = 100000  # learning rate
    simulation_name = "autograd_short_phi42_a05"
    path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}/"
    uuid_name = "sand2d_inverse_eval"
    mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
    analysis_dt = 1e-06
    output_steps = 2500
    analysis_nsteps = 2500*5 + 1  # only run to get 6 initial positions to make X_0 in GNS

    # inputs for make `.npz` containing initial 6 steps. This npz file corresponds to X_0
    material_feature = True
    ndim = 2
    dt = 1.0

    # inputs for forward rollout
    device = torch.device('cpu')
    noise_std = 6.7e-4  # hyperparameter used to train GNS.
    NUM_PARTICLE_TYPES = 9
    model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-r015/"
    model_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-r015/"
    model_file = "model-5000000.pt"
    nforward_steps = 30
    target_step = nforward_steps

    # inputs for render
    xbound = [-0.0025, 1.0025]
    ybound = [-0.0025, 0.5025]

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
        resume_epoch = records[-1]["epoch"]
        print(f"Start optimization from phi: {phi}, epoch: {resume_epoch}")


    # Start forward computations and gradient descent
    for epoch in range(nepochs):
        if resume:
            epoch = epoch + resume_epoch + 1
        else:
            pass
        print(f"Epoch {epoch}---------------------------------------------------")
        start_epoch = time.time()

        # start recording inverse analysis information for current phi
        record = {}
        record["epoch"] = epoch
        record["learning_rate"] = lr
        record["current_phi"] = phi

        # RUN MPM FOR CURRENT PHI GUESS
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
        uuid = f"/results/{uuid_name}"
        convert_hd5_to_npz(path=path + sim_name,
                           uuid=uuid,
                           ndim=ndim,
                           output=f"{path}{sim_name}/{sim_name}.npz",
                           material_feature=material_feature,
                           dt=dt
                           )

        # %% ROLLOUT FOR EACH PHI GUESS
        metadata = reading_utils.read_metadata(model_metadata_path)
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
            target_final_runout = np.max(target_final_position[:, 0])

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
        loss, predicted_final_position = forward_rollout_autograd(
            simulator, initial_positions, particle_type, phi_guess,
            n_particles_per_example, nforward_steps, target_final_runout,
            device)
        print(f"Loss: {loss}")

        end_rollout = time.time()
        rollout_time = end_rollout - start_rollout
        print(f"Rollout for phi {phi} took {rollout_time}s")
        record[f"rollout_time"] = rollout_time

        # save current prediction
        filename = f'{path}/mpm_phi{phi}/mpm_phi{phi}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(predicted_final_position, f)

        start_backprop = time.time()
        loss.to(device).requires_grad_(True)
        loss.backward(retain_graph=True, inputs=[phi_guess])
        loss.detach()

        # Access gradients of input material_type
        grads = phi_guess.grad.cpu().detach().numpy()
        end_backprop = time.time()
        backprop_time = end_backprop - start_backprop
        print(f"Grad computation at {phi} took {backprop_time}s")
        record[f"backprop_time"] = backprop_time


        # # Get full rollout just for render purpose
        # device = torch.device('cuda')
        # simulator.to(device)
        # simulator.eval()
        # with torch.no_grad():
        #     for example_i, features in enumerate(dinit):  # TODO: only one item exists in `dint`. No need for loop
        #         # Obtain features
        #         if len(features) < 3:
        #             raise NotImplementedError("Data should include material feature")
        #         positions = features[0].to(device).requires_grad_(False)
        #         particle_type = features[1].to(device).requires_grad_(False)
        #         phi_torch = features[2][0].requires_grad_(False)
        #         n_particles_per_example = torch.tensor(
        #             [int(features[3])], dtype=torch.int32).to(device).requires_grad_(False)
        #
        #         # Compute runout loss
        #         start_full_rollout = time.time()
        #         target_full_final_runout = info[0][-1]
        #         print(f"Compute full rollout at phi: {phi}")
        #         full_rollout_data, full_rollout_loss = forward_rollout(
        #             simulator, positions, particle_type, phi_torch,
        #             n_particles_per_example, metadata["sequence_length"], target_full_final_runout,
        #             device
        #         )
        #         end_full_rollout = time.time()
        #         full_rollout_time = end_full_rollout - start_full_rollout
        #         print(f"Rollout for phi {phi} took {full_rollout_time}s")
        #
        # record["full_rollout"] = full_rollout_data
        # record["full_rollout_loss"] = full_rollout_loss
        # record["full_rollout_time"] = full_rollout_time

        # update phi
        phi = phi - lr * grads
        print(f"Gradient of loss w.r.t phi is: {grads}")

        end_epoch = time.time()
        print(f"Epoch {epoch} took {end_epoch - start_epoch}s")


        # record rollout information
        record["loss"] = loss.cpu().detach().numpy()
        record["rollout"] = {
            "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
            "predicted_final_positions": predicted_final_position,
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

        # save plot for predicted final position
        plot_final_position(path=f'{path}/mpm_phi{record["current_phi"]}/mpm_phi{record["current_phi"]}.pkl',
                            output=f'{path}/mpm_phi{record["current_phi"]}/',
                            xbound=xbound,
                            ybound=ybound)


        # # free memory
        # free_gpu_cache()
        # simulator = simulator.to("cpu")
        del simulator
        del grads
        del loss
        del predicted_final_position



if __name__ == '__main__':

    autograd_inverse()