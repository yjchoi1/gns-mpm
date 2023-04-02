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
from convert_hd5_to_npz import convert_hd5_to_npz
from animation_from_pkl import animation_from_pkl
from absl import flags
from absl import app

# inputs
resume = False
nepoch = 20
lr = 100000  # learning rate
simulation_name = "sand2d_frictions_test1"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/{simulation_name}/"
phi = 45  # initial guess of phi
dphi = 1  # Delta_phi for finite difference to compute gradient of loss (dJ/dphi) where J = X_final - X'_final
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

# inputs for render
xbound = [-0.0025, 1.0025]
ybound = [-0.0025, 1.0025]

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
    phi = records[-1]["phi"]

# Start forward computations and gradient descent
start_inverse = time.time()
for epoch in range(nepoch):
    start_epoch = time.time()

    # loss (X_final - X'_final) obtained from phi and phi+dphi
    loss_between_dphi = []

    # start recording inverse analysis information for current phi
    record = {}
    record[f"epoch{epoch}"] = epoch

    # run mpm for two datapoints (i.e., phi & phi+dphi)
    for i, guess in enumerate([phi, phi + dphi]):

        record["phi"] = phi

        # %% PREPARE MPM INPUT FILE FOR EACH PHI
        # open initial mpm input with arbitrary phi value
        f = open(os.path.join(path, mpm_input))
        # modify phi in the mpm input file
        mpm_input_guess = json.load(f)
        mpm_input_guess["materials"][0]["friction"] = guess
        mpm_input_guess["materials"][0]["residual_friction"] = guess
        mpm_input_guess["analysis"]["dt"] = analysis_dt
        mpm_input_guess["analysis"]["nsteps"] = analysis_nsteps
        mpm_input_guess["ouput_steps"] = output_steps
        f.close()

        # %% PREPARE MPM INPUTS
        # make mpm folder for current phi guess
        mpm_folder = os.path.join(path, f"mpm_phi{guess}")
        os.makedirs(mpm_folder, exist_ok=True)
        # mpm input files should be located in the `path`
        file_paths = [
            f'{path}/mpm_metadata.json',
            f'{path}/mesh.txt',
            f'{path}/particles.txt',
            f'{path}/entity_sets.json',
            f'{path}/initial_config.png'
        ]
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f'{file_path} not found')
        # copy mpm inputs to mpm folder for current phi guess
        shutil.copyfile(f'{path}/mpm_metadata.json', f'{mpm_folder}/metadata.json')
        shutil.copyfile(f'{path}/mesh.txt', f'{mpm_folder}/mesh.txt')
        shutil.copyfile(f'{path}/particles.txt', f'{mpm_folder}/particles.txt')
        shutil.copyfile(f'{path}/entity_sets.json', f'{mpm_folder}/entity_sets.json')
        shutil.copyfile(f'{path}/initial_config.png', f'{mpm_folder}/initial_config.png')
        # make mpm input file for phi guess
        with open(os.path.join(mpm_folder, f"mpm_input.json"), 'w') as f:
            json.dump(mpm_input_guess, f, indent=2)
        f.close()

        # record current mpm input
        record[f"mpm_input_files_phi{guess}"] = mpm_input_guess

        # %% RUN MPM
        start_mpm = time.time()
        # write bash file to run mpm
        with open(f'{path}/run_mpm.sh', 'w') as rsh:
            rsh.write(
                f'''\
                #! /bin/bash
                module reset
                /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "{path}/mpm_phi{guess}/"
                ''')

        # run mpm
        print(f"Running MPM for {phi} at epoch {epoch}...")
        with open(f'{path}/run_mpm.sh', 'rb') as bashfile:
            script = bashfile.read()
        with open(f"{mpm_folder}/mpm_out.txt", 'w') as outfile:
            rc = subprocess.call(script, shell=True, stdout=outfile)

        end_mpm = time.time()
        print(f"MPM for phi {guess} took {end_mpm - start_mpm}s")

        # make `.npz` to prepare initial state X_1 for rollout
        sim_name = f"mpm_phi{guess}"
        uuid = "/results/sand2d_frictions_test"
        convert_hd5_to_npz(path=path + sim_name,
                           uuid=uuid,
                           ndim=ndim,
                           output=f"{path}{sim_name}/{sim_name}.npz",
                           material_feature=material_feature,
                           dt=dt
                           )

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
        with torch.no_grad():
            for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need for loop
                # Obtain features
                if len(features) < 3:
                    raise NotImplementedError("Data should include material feature")
                positions = features[0].to(device).requires_grad_(False)
                particle_type = features[1].to(device).requires_grad_(False)
                phi_torch = features[2][0].requires_grad_(False)
                n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device).requires_grad_(
                    False)

                # Compute runout loss
                start_rollout = time.time()
                print(f"Compute rollout at phi: {guess}")
                rollout_data, loss = forward_rollout(
                    simulator, positions, particle_type, phi_torch,
                    n_particles_per_example, metadata["sequence_length"], target_final_position,
                    device
                )
                end_rollout = time.time()
                print(f"Rollout for phi {guess} took {end_rollout - start_rollout}s")

                print(f"Loss at {guess}: {loss.cpu().numpy()}")
                loss_between_dphi.append(loss.cpu().numpy())

                # Save rollout in testing
                rollout_data['metadata'] = metadata
                rollout_data['loss'] = loss.cpu().numpy()
                filename = f'{path}/mpm_phi{guess}/mpm_phi{guess}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(rollout_data, f)

        # see how the rollout looks like by animation
        render_start = time.time()
        animation_from_pkl(path=f'{path}/mpm_phi{guess}/mpm_phi{guess}.pkl',
                           output=f'{path}/mpm_phi{guess}/',
                           xbound=xbound,
                           ybound=ybound)
        render_end = time.time()
        print(f"Render for phi {guess} took {render_end - render_start}s")

        # record rollout and loss
        record[f"rollout_phi{guess}"] = rollout_data
        record[f"loss_phi{guess}"] = loss.cpu().numpy()

    # %% RADIENT DESCENT AND UPDATE PHI
    grad = (loss_between_dphi[0] - loss_between_dphi[1]) / (phi - (phi + dphi))
    phi = phi - lr * grad  # TODO: what is good gradient scaling method? Current value is too small
    record["grad"] = grad
    record["next_guess"] = phi

    # append current epoch's record
    records.append(record)

    # save current record file
    with open(record_file, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    end_epoch = time.time()
    print(f"Epoch {epoch} took {end_epoch - start_epoch}s")

end_inverse = time.time()
print(f"Inversion for {nepoch} {end_inverse - start_inverse}s")
