import torch
import time
import os
import sys
import numpy as np
import json
import glob
from utils import Make_it_to_torch_model
from io import StringIO
from matplotlib import pyplot as plt
# from utils import render
import torch.utils.checkpoint
from forward import rollout_with_checkpointing
from utils import make_animation
from utils import visualize_final_deposits
from utils import visualize_velocity_profile

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import reading_utils
from gns import data_loader
from gns import train


# inputs
resume = False
resume_epoch = 1

nepoch = 50
inverse_timestep_range = [300, 380]
checkpoint_interval = 1
lr = 0.1
simulation_name = "multivar2_inverse"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/{simulation_name}/"
ground_truth_npz = "sand2d_inverse_eval29.npz"
ground_truth_mpm_inputfile = "mpm_input.json"
dt_mpm = 0.0025

# inputs for forward simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/"
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/"
model_file = "model-7020000.pt"

# outputs
output_dir = f"/outputs/"
save_step = 1

#%%

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)["rollout"]
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth particle position at the inversion timestep
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
target_final_positions = torch.tensor(
    mpm_trajectory[0][0][inverse_timestep_range[0]: inverse_timestep_range[1]], device=device)

# Get ground truth velocities for each particle group.
f = open(f"{path}/{ground_truth_mpm_inputfile}")
mpm_inputs = json.load(f)
velocity_constraints = mpm_inputs["mesh"]["boundary_conditions"]["particles_velocity_constraints"]
# Initialize an empty NumPy array with the shape (max_pset_id+1, 2)
max_pset_id = max(item['pset_id'] for item in velocity_constraints)
ground_truth_vels = np.zeros((max_pset_id + 1, 2))
# Fill in the NumPy array with velocity values from data
for constraint in velocity_constraints:
    pset_id = constraint['pset_id']
    dir = constraint['dir']
    velocity = constraint['velocity']
    ground_truth_vels[pset_id, dir] = velocity


# Get initial position (i.e., p_0) for each particle group
# TODO (yc): improve indexing
# particle_files = sorted(glob.glob(f"{path}/particles*.txt"))
# particle_groups = []
# particle_groups_idxs = []
# count = 0
# for filename in particle_files:
#     particle_group = torch.tensor(np.loadtxt(filename, skiprows=1))
#     particle_groups.append(particle_group)
#     indices = np.arange(count, count+len(particle_group))
#     count = count+len(particle_group)
#     particle_groups_idxs.append(indices)
# initial_position = torch.concat(particle_groups).to(device)

particle_files = sorted(glob.glob(f"{path}/particles*.txt"))
particle_groups = []
particle_group_idx_ranges = []
count = 0
for filename in particle_files:
    particle_group = torch.tensor(np.loadtxt(filename, skiprows=1))
    particle_groups.append(particle_group)
    particle_group_idx_range = np.arange(count, count+len(particle_group))
    count = count+len(particle_group)
    particle_group_idx_ranges.append(particle_group_idx_range)
initial_position = torch.concat(particle_groups).to(device)

# Initialize initial velocity (i.e., dot{p}_0)
initial_velocity = torch.tensor(
    [[0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0],
     [0.0, 0.0]],
    requires_grad=True, device=device)
initial_velocity_model = Make_it_to_torch_model(initial_velocity)

# Set up the optimizer
optimizer = torch.optim.Adam(initial_velocity_model.parameters(), lr=lr)

# Set output folder
if not os.path.exists(f"{path}/{output_dir}"):
    os.makedirs(f"{path}/{output_dir}")

# Resume
if resume:
    print(f"Resume from the previous state: epoch{resume_epoch}")
    checkpoint = torch.load(f"{path}/{output_dir}/optimizer_state-{resume_epoch}.pt")
    start_epoch = checkpoint["epoch"]
    initial_velocity_model.load_state_dict(checkpoint['velocity_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_epoch = 0
initial_velocity = initial_velocity_model.current_params

# Start optimization iteration
for epoch in range(start_epoch+1, nepoch):
    start = time.time()
    optimizer.zero_grad()  # Clear previous gradients

    # Load data containing X0, and get necessary features.
    # First, get particle type and material property.
    dinit = data_loader.TrajectoriesDataset(path=f"{path}/{ground_truth_npz}")
    for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need `for` loop
        # Obtain features
        if len(features) < 3:
            raise NotImplementedError("Data should include material feature")
        particle_type = features[1].to(device)
        material_property = features[2].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)

    # Next, make [p0, p1, ..., p5] using current initial velocity, assuming that velocity is the same for 5 timesteps
    initial_pos_seq_all_group = []
    for i, particle_group_idx_range in enumerate(particle_group_idx_ranges):
        initial_pos_seq_each_group = [
            initial_position[particle_group_idx_range] + initial_velocity[i] * dt_mpm * t for t in range(6)]
        initial_pos_seq_all_group.append(torch.stack(initial_pos_seq_each_group))
    initial_positions = torch.concat(initial_pos_seq_all_group, axis=1).to(device).permute(1, 0, 2).to(torch.float32).contiguous()

    print(f"Initial velocities: {initial_velocity.detach().cpu().numpy()}")

    predicted_positions = rollout_with_checkpointing(
        simulator=simulator,
        initial_positions=initial_positions,
        particle_types=particle_type,
        material_property=material_property,
        n_particles_per_example=n_particles_per_example,
        nsteps=inverse_timestep_range[1] - initial_positions.shape[1] + 1, # exclude initial positions (x0) which we already have
        checkpoint_interval=checkpoint_interval
    )

    inversion_positions = predicted_positions[inverse_timestep_range[0]:inverse_timestep_range[1]]

    loss = torch.mean((inversion_positions - target_final_positions) ** 2)
    print("Backpropagate...")
    loss.backward()

    # Visualize current prediction
    print(f"Epoch {epoch - 1}, Loss {loss.item():.8f}")
    print(f"Initial vel: {initial_velocity.detach().cpu().numpy()}")
    visualize_final_deposits(predicted_positions=predicted_positions,
                             target_positions=target_final_positions,
                             metadata=metadata,
                             write_path=f"{path}/{output_dir}/final_deposit-{epoch - 1}.png")
    visualize_velocity_profile(predicted_velocities=initial_velocity,
                               target_velocities=ground_truth_vels,
                               write_path=f"{path}/{output_dir}/vel_profile-{epoch - 1}.png")

    # Perform optimization step
    optimizer.step()

    end = time.time()
    time_for_iteration = end - start

    # Save and report optimization status
    if epoch % save_step == 0:

        # Make animation at the last epoch
        if epoch == nepoch - 1:
            print(f"Rendering animation at {epoch}...")
            positions_np = np.concatenate(
                (initial_positions.permute(1, 0, 2).detach().cpu().numpy(),
                 predicted_positions.detach().cpu().numpy())
            )
            make_animation(positions=positions_np,
                           boundaries=metadata["bounds"],
                           output=f"{path}/{output_dir}/animation-{epoch}.gif",
                           timestep_stride=5)

        # Save history
        current_history = {
            "epoch": epoch,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
            "initial_velocity": initial_velocity.detach().cpu().numpy(),
            "loss": loss.item()
        }

        # Save optimizer state
        torch.save({
            'epoch': epoch,
            'time_spent': time_for_iteration,
            'position_state_dict': {
                "target_positions": mpm_trajectory[0][0],
                "inversion_positions": predicted_positions.clone().detach().cpu().numpy()
            },
            'velocity_state_dict': Make_it_to_torch_model(initial_velocity).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"{path}/{output_dir}/optimizer_state-{epoch}.pt")
        a = 1
        # # Save optimizer state
        # torch.save({
        #     'epoch': epoch,
        #     'position_state_dict': {
        #         "target_positions": mpm_trajectory[0][0], "inversion_positions": predicted_positions
        #     },
        #     'initial_velocity': Make_it_to_torch_model(initial_velocity).state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss
        # }, f"{path}/outputs/optimizer_state-{epoch}.pt")
