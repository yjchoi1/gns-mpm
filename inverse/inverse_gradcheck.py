import torch
import time
import os
import sys
import numpy as np
import glob
from utils import To_Torch_Model_Param
from matplotlib import pyplot as plt
import torch.utils.checkpoint
from forward import rollout_with_checkpointing
from utils import make_animation
from utils import compute_penalty
from utils import visualize_final_deposits
from run_mpm import run_mpm

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import reading_utils
from gns import data_loader
from gns import train
from convert_hd5_to_npz import convert_hd5_to_npz


# inputs
resume = False
resume_epoch = 14

diff_method = "fd"  # ad or fd
x0_mode = "from_same_5vels"  # `from_mpm` or `from_same_5vels`
nepoch = 21
if diff_method == "fd":
    dphi = 0.05  # just for fd

# inputs for MPM to make X0 (i.e., p0, p1, p2, p3, p4, p5)
if x0_mode == "from_mpm":
    uuid_name = "sand2d_inverse_eval"
    mpm_input = "mpm_input.json"  # mpm input file to start running MPM for phi & phi+dphi
    analysis_dt = 1e-06
    output_steps = 2500
    analysis_nsteps = 2500 * 5 + 1  # only run to get 6 initial positions to make X_0 in GNS
    ndim = 2

inverse_timestep = 379
checkpoint_interval = 1
lr = 500  # learning rate (phi=21: 1000, phi=42: 3000)
simulation_name = "tall_phi42"
path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/{simulation_name}/"
ground_truth_npz = "sand2d_inverse_eval28.npz"
phi = 30.0  # initial guess of phi
loss_constraint = True
if loss_constraint == True:
    loss_limit = 0.0005
    penalty_mag = 1

# inputs for forward simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-sr020/"
simulator_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand2d_frictions-sr020/"
model_file = "model-7020000.pt"

# outputs
output_dir = f"/outputs_{diff_method}_const_lim{loss_limit}_mag{penalty_mag}_lr{lr}/"
save_step = 1

# ---------------------------------------------------------------------------------

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout")
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
friction_model = To_Torch_Model_Param(friction)

# Set up the optimizer
optimizer = torch.optim.SGD(friction_model.parameters(), lr=lr)

# Set output folder
if not os.path.exists(f"{path}/{output_dir}"):
    os.makedirs(f"{path}/{output_dir}")

# Resume
if resume:
    print(f"Resume from the previous state: epoch{resume_epoch}")
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

    if x0_mode == "from_mpm":
        # Run MPM with current friction angle to get X0
        run_mpm(path,
                output_dir,
                mpm_input,
                epoch,
                friction.item(),
                analysis_dt,
                analysis_nsteps,
                output_steps)

        # Make `.npz` to prepare initial state X_1 for rollout
        convert_hd5_to_npz(path=f"{path}/{output_dir}/mpm_epoch-{epoch}/",
                           uuid=f"/results/{uuid_name}/",
                           ndim=ndim,
                           output=f"{path}/{output_dir}/x0_epoch-{epoch}.npz",
                           material_feature=True,
                           dt=1.0)

    # Load data containing X0, and get necessary features.
    # First, obtain ground truth features except for material property
    if x0_mode == "from_mpm":
        dinit = data_loader.TrajectoriesDataset(path=f"{path}/{output_dir}/x0_epoch-{epoch}.npz")
    if x0_mode == "from_same_5vels":
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
        material_property_tensor = torch.tan((friction * torch.pi / 180)) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

        print("Start rollout...")
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=initial_positions,
            particle_types=particle_type,
            material_property=material_property_tensor,
            n_particles_per_example=n_particles_per_example,
            nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
            checkpoint_interval=checkpoint_interval,
        )

        inversion_runout = predicted_positions[inverse_timestep, :, 0].max()

        loss = (inversion_runout - target_final_runout) ** 2
        if loss_constraint:
            penalty = compute_penalty(loss, threshold=loss_limit, alpha=penalty_mag)
            loss = loss + penalty

        print("Backpropagate...")
        loss.backward()

    elif diff_method == "fd":  # finite diff
        # Prepare (phi, phi+dphi)
        material_property_tensor = torch.tan((friction * torch.pi / 180)) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()
        material_property_tensor_perturb = torch.tan((friction+dphi) * torch.pi / 180) * torch.full(
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
        loss = (inversion_runout - target_final_runout) ** 2
        loss_perturb = (inversion_runout_perturb - target_final_runout) ** 2
        gradient = (loss_perturb - loss) / dphi
        friction.grad = torch.tensor([gradient], dtype=friction.dtype, device=friction.device)
    else:
        raise NotImplementedError

    # Visualize current prediction
    print(f"Epoch {epoch-1}, Friction {friction.item():.5f}, Loss {loss.item():.8f}")
    visualize_final_deposits(friction.item(),
                             predicted_positions,
                             target_positions,
                             metadata,
                             write_path=f"{path}/{output_dir}/inversion-{epoch-1}.png")

    # Perform optimization step
    optimizer.step()

    end = time.time()
    time_for_iteration = end - start

    # Save and report optimization status
    if epoch % save_step == 0:

        # Make animation at the last epoch
        if epoch == nepoch-1:
            print(f"Rendering animation at {epoch}...")
            positions_np = np.concatenate(
                (initial_positions.permute(1, 0, 2).detach().cpu().numpy(),
                 predicted_positions.detach().cpu().numpy())
            )
            make_animation(positions=positions_np,
                           boundaries=metadata["bounds"],
                           output=f"{path}/{output_dir}/animation-{epoch}.gif",
                           timestep_stride=5)

        # Save optimizer state
        torch.save({
            'epoch': epoch,
            'time_spent': time_for_iteration,
            'position_state_dict': {
                "target_positions": predicted_positions.clone().detach().cpu().numpy(),
                "inversion_positions": target_positions.clone().detach().cpu().numpy()
            },
            'friction_state_dict': Make_it_to_torch_model(friction).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_constraints': {"loss_limit": loss_limit, "penalty_mag": penalty_mag} if loss_constraint else None,
            'dphi': dphi if diff_method == "fd" else None
        }, f"{path}/{output_dir}/optimizer_state-{epoch}.pt")