import torch
import time
import os
import sys
import numpy as np
import glob
from utils.torch_model_wrapper import Make_it_to_torch_model
from io import StringIO
from matplotlib import pyplot as plt
from utils import render
import torch.utils.checkpoint

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from forward import forward_rollout_velocity, rollout_with_checkpointing
from gns import reading_utils
from gns import data_loader
from gns import train


# inputs
simulation_name = "multivar1_inverse"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/{simulation_name}/"
ground_truth_npz = "multivar1.npz"
dt_mpm = 0.0025
inverse_timestep = 200
nepoch = 300
lr = 10
resume = False
resume_epoch = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
history_print_step = 10
save_step = 10

gns_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-r015/"
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-r015/"
model_file = "model-8890000.pt"
noise_std = 6.7e-4  # hyperparameter used to train GNS.


# Load simulator
metadata = reading_utils.read_metadata(gns_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth particle position at the inversion timestep
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
target_final_positions = torch.tensor(mpm_trajectory[0][0][inverse_timestep - 6:inverse_timestep], device=device)

# Get initial position (i.e., p_0) for each particle group
particle_files = sorted(glob.glob(f"{path}/particles*.txt"))
particle_groups = []
particle_groups_idxs = []
count = 0
for filename in particle_files:
    particle_group = torch.tensor(np.loadtxt(filename, skiprows=1))
    particle_groups.append(particle_group)
    indices = np.arange(count, count+len(particle_group))
    count = count+len(particle_group)
    particle_groups_idxs.append(indices)
initial_position = torch.concat(particle_groups).to(device)

# Initialize initial velocity (i.e., dot{p}_0)
initial_velocity = torch.tensor(
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    requires_grad=True, device=device)

# Set up the optimizer
optimizer = torch.optim.Adam([initial_velocity], lr=lr)
initial_velocity_model = Make_it_to_torch_model(initial_velocity)

# Set output folder
if not os.path.exists(f"{path}/outputs/"):
    os.makedirs(f"{path}/outputs/")

# Resume
if resume:
    print("Resume from the previous state")
    checkpoint = torch.load(f"{path}/outputs/optimizer_state-{resume_epoch}.pt")
    start_epoch = checkpoint["epoch"]
    initial_velocity_model.load_state_dict(checkpoint['friction_state_dict'])
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
    initial_positions_all_group = []
    for i, particle_group_idxs in enumerate(particle_groups_idxs):
        initial_positions_each_group = [
            initial_position[particle_group_idxs] + initial_velocity[i] * dt_mpm * t for t in range(6)]
        initial_positions_all_group.append(torch.stack(initial_positions_each_group))
    initial_positions = torch.concat(initial_positions_all_group, axis=1).to(device).permute(1, 0, 2).to(torch.float32).contiguous()

    if epoch % save_step == 0:
        print(f"Initial velocities: {initial_velocity[0].detach().cpu().numpy(), initial_velocity[1].detach().cpu().numpy(), initial_velocity[2].detach().cpu().numpy()}")

    predicted_positions = rollout_with_checkpointing(
        simulator=simulator,
        initial_positions=initial_positions,
        particle_types=particle_type,
        material_property=material_property,
        n_particles_per_example=n_particles_per_example,
        nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
    )

    inversion_positions = predicted_positions[inverse_timestep - 6:inverse_timestep]

    loss = torch.mean((inversion_positions - target_final_positions) ** 2)
    loss.backward()
    optimizer.step()

    end = time.time()
    time_for_iteration = end - start

    # Save and report optimization status
    if epoch % save_step == 0:
        # print status
        print(f"Epoch {epoch}, Loss {loss.item()}")
        print(f"Initial velocities: {initial_velocity[0].detach().cpu().numpy(), initial_velocity[1].detach().cpu().numpy(), initial_velocity[2].detach().cpu().numpy()}")

        # visualize and save inversion state
        fig, ax = plt.subplots()
        inversion_position_plot = inversion_positions.clone().detach().cpu().numpy()
        target_final_position_plot = target_final_positions.clone().detach().cpu().numpy()
        ax.scatter(inversion_position_plot[-1][:, 0],
                   inversion_position_plot[-1][:, 1], alpha=0.5, s=0.2, c="gray", label="Current")
        ax.scatter(target_final_position_plot[-1][:, 0],
                   target_final_position_plot[-1][:, 1], alpha=0.5, s=0.2, c="red", label="True")
        ax.set_xlim(metadata['bounds'][0])
        ax.set_ylim(metadata['bounds'][1])
        ax.legend()
        ax.grid(True)
        plt.savefig(f"{path}/outputs/inversion-{epoch}.png")

        # Animation
        print(f"Rendering animation at {epoch}...")
        positions_np = np.concatenate(
            (initial_positions.permute(1, 0, 2).detach().cpu().numpy(),
             predicted_positions.detach().cpu().numpy())
        )
        render.make_animation(positions=positions_np,
                              boundaries=metadata["bounds"],
                              output=f"{path}/outputs/animation-{epoch}.gif",
                              timestep_stride=3)

        # Save history
        print(f"Rendering animation at {epoch}...")
        current_history = {
            "epoch": epoch,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
            "initial_velocity": initial_velocity.detach().cpu().numpy(),
            "loss": loss.item()
        }

        # Save optimizer state
        torch.save({
            'epoch': epoch,
            'position_state_dict': {
                "target_positions": mpm_trajectory[0][0], "inversion_positions": predicted_positions
            },
            'initial_velocity': Make_it_to_torch_model(initial_velocity).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"{path}/outputs/optimizer_state-{epoch}.pt")
