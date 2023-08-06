import torch
import time
import os
import sys
import numpy as np
import glob
from utils.torch_model_wrapper import Make_it_to_torch_model
from matplotlib import pyplot as plt
import torch.utils.checkpoint
from forward import rollout_with_checkpointing
from utils import render
from run_mpm import run_mpm

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import reading_utils
from gns import data_loader
from gns import train
from convert_hd5_to_npz import convert_hd5_to_npz


# inputs
resume = False
resume_epoch = 10

diff_method = "ad"  # ad or fd
dphi = 0.05
nepoch = 11

inverse_timestep = 379
checkpoint_interval = 1
lr = 1000  # learning rate
simulation_name = "tall_phi21"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/{simulation_name}/"
ground_truth_npz = "sand2d_inverse_eval26.npz"
phi = 30.0  # initial guess of phi

# inputs for MPM to make X0 (i.e., p0, p1, p2, p3, p4, p5)
uuid_name = "sand2d_inverse_eval"
mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
analysis_dt = 1e-06
output_steps = 2500
analysis_nsteps = 2500 * 5 + 1  # only run to get 6 initial positions to make X_0 in GNS
ndim = 2

# inputs for forward rollout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/"
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/"
model_file = "model-6300000.pt"

# outputs
output_dir = f"/outputs_{diff_method}/"
save_step = 1

# ---------------------------------------------------------------------------------

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth particle position at the inversion timestep
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
target_positions = torch.tensor(mpm_trajectory[0][0])
target_final_runout = target_positions[inverse_timestep][:, 0].max().to(device)

# Initialize friction angle to start optimizing
friction = torch.tensor([phi], requires_grad=True, device=device)
friction_model = Make_it_to_torch_model(friction)

# Set up the optimizer
optimizer = torch.optim.SGD(friction_model.parameters(), lr=lr)

# Set output folder
if not os.path.exists(f"{path}/{output_dir}"):
    os.makedirs(f"{path}/{output_dir}")

# Resume
if resume:
    print("Resume from the previous state")
    checkpoint = torch.load(f"{path}/{output_dir}/optimizer_state-{resume_epoch}.pt")
    start_epoch = checkpoint["epoch"]
    friction_model.load_state_dict(checkpoint['friction_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_epoch = 0
friction = friction_model.current_params

# Start optimization iteration
for epoch in range(start_epoch+1, nepoch):
    start = time.time()
    optimizer.zero_grad()  # Clear previous gradients

    # # Run MPM with current friction angle to get X0
    # run_mpm(path,
    #         mpm_input,
    #         epoch,
    #         friction.item(),
    #         analysis_dt,
    #         analysis_nsteps,
    #         output_steps)
    #
    # # Make `.npz` to prepare initial state X_1 for rollout
    # convert_hd5_to_npz(path=f"{path}/{output_dir}/mpm_epoch-{epoch}/",
    #                    uuid=f"/results/{uuid_name}/",
    #                    ndim=ndim,
    #                    output=f"{path}/{output_dir}/x0_epoch-{epoch}.npz",
    #                    material_feature=True,
    #                    dt=1.0)

    # Load data containing X0, and get necessary features.
    # First, obtain ground truth features except for material property
    # dinit = data_loader.TrajectoriesDataset(path=f"{path}/{output_dir}/x0_epoch-{epoch}.npz")
    dinit = data_loader.TrajectoriesDataset(path=f"{path}/{ground_truth_npz}")
    for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need `for` loop
        if len(features) < 3:
            raise NotImplementedError("Data should include material feature")
        initial_positions = features[0][:, :6, :].to(device)
        particle_type = features[1].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)

    # Do forward pass as compute gradient of parameter
    if diff_method == "ad":
        # Make material property feature from current phi
        material_property_tensor = (friction * torch.pi / 180) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

        print("Start rollout...")
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=initial_positions,
            particle_types=particle_type,
            material_property=material_property_tensor,
            n_particles_per_example=n_particles_per_example,
            nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
            checkpoint_interval=checkpoint_interval
        )

        inversion_runout = predicted_positions[inverse_timestep, :, 0].max()

        loss = torch.mean((inversion_runout - target_final_runout) ** 2)
        print("Backpropagate...")
        loss.backward()

    elif diff_method == "fd":  # finite diff
        # Prepare (phi, phi+dphi)
        material_property_tensor = (friction * torch.pi / 180) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()
        material_property_tensor_perturb = ((friction+dphi) * torch.pi / 180) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

        # Rollout at [phi & phi+dphi]
        with torch.no_grad():
            print("Start rollout with phi")
            predicted_positions = rollout_with_checkpointing(
                simulator=simulator,
                initial_positions=initial_positions,
                particle_types=particle_type,
                material_property=material_property_tensor,
                n_particles_per_example=n_particles_per_example,
                nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
                checkpoint_interval=checkpoint_interval,
                is_checkpointing=False
            )
            print("Start rollout with phi+dphi")
            predicted_positions_perturb = rollout_with_checkpointing(
                simulator=simulator,
                initial_positions=initial_positions,
                particle_types=particle_type,
                material_property=material_property_tensor_perturb,
                n_particles_per_example=n_particles_per_example,
                nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
                checkpoint_interval=checkpoint_interval,
                is_checkpointing=False
            )

        # Compute gradient of loss: (loss(phi+dphi) - loss(phi))/dphi
        inversion_runout = predicted_positions[inverse_timestep, :, 0].max()
        inversion_runout_perturb = predicted_positions_perturb[inverse_timestep, :, 0].max()
        loss = torch.mean((inversion_runout - target_final_runout) ** 2)
        loss_perturb = torch.mean((inversion_runout_perturb - target_final_runout) ** 2)
        gradient = (loss_perturb - loss) / dphi
        friction.grad = torch.tensor([gradient], dtype=friction.dtype, device=friction.device)
    else:
        raise NotImplementedError

    # Perform optimization step
    optimizer.step()

    end = time.time()
    time_for_iteration = end - start

    # Save and report optimization status
    if epoch % save_step == 0:
        # print status
        print(f"Epoch {epoch}, Friction {friction.item():.5f}, Loss {loss.item():.8f}")

        # visualize and save inversion state
        fig, ax = plt.subplots()
        inversion_positions_plot = predicted_positions.clone().detach().cpu().numpy()
        target_positions_plot = target_positions.clone().detach().cpu().numpy()
        ax.scatter(inversion_positions_plot[-1][:, 0],
                   inversion_positions_plot[-1][:, 1], alpha=0.5, s=2.0, c="purple", label="Current")
        plt.axvline(x=inversion_positions_plot[-1][:, 0].max(), c="purple")
        ax.scatter(target_positions_plot[-1][:, 0],
                   target_positions_plot[-1][:, 1], alpha=0.5, s=2.0, c="yellow", label="True")
        plt.axvline(x=target_positions_plot[-1][:, 0].max(), c="yellow")
        ax.set_xlim(metadata['bounds'][0])
        ax.set_ylim(metadata['bounds'][1])
        ax.legend()
        ax.grid(True)
        plt.savefig(f"{path}/{output_dir}/inversion-{epoch}.png")

        # Make animation at the last epoch
        if epoch == nepoch-1:
            print(f"Rendering animation at {epoch}...")
            positions_np = np.concatenate(
                (initial_positions.permute(1, 0, 2).detach().cpu().numpy(),
                 predicted_positions.detach().cpu().numpy())
            )
            render.make_animation(positions=positions_np,
                                  boundaries=metadata["bounds"],
                                  output=f"{path}/{output_dir}/animation-{epoch}.gif",
                                  timestep_stride=5)

        # Save optimizer state
        torch.save({
            'epoch': epoch,
            'time_spent': time_for_iteration,
            'position_state_dict': {
                "target_positions": inversion_positions_plot, "inversion_positions": inversion_positions_plot
            },
            'friction_state_dict': Make_it_to_torch_model(friction).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'dphi': dphi if diff_method == "fd" else None
        }, f"{path}/{output_dir}/optimizer_state-{epoch}.pt")