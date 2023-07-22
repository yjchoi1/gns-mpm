import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle
from forward import rollout
import utils

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/inverse-flow')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/meshnet/')
from meshnet import learned_simulator
from meshnet.utils import NodeType


# Inputs
resume = False
simulation_name = "trj1"
path = f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/inverse/prac/{simulation_name}/"
ground_truth_npz = "trj1.npz"
model_file = "model-10000000.pt"
train_state_file = "train_state-10000000.pt"
inverse_timestep = 21
simulation_region = [[0.0, 1.6], [0.0, 0.4]]
offset = 0.01  # offset from the simulation_region to place obstacle
obstacle_region = [[simulation_region[0][0]+offset, simulation_region[0][1]-offset],
                   [simulation_region[1][0]+offset, simulation_region[1][1]-offset]]
vel_observe_region = [[0.5, 0.6], [0.0, 4.0]]
radius_tol = 0.00001
# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
history_print_step = 1
save_step = 1
# Optimization params
nepoch = 30
lr = 0.01
# Finite difference hyperparameter
fd_perturb = [torch.tensor([0.02, 0.0, 0.0]),
              torch.tensor([0.0, 0.02, 0.0]),
              torch.tensor([0.0, 0.0, 0.01])]


# Import data
data = [dict(trj_info.item()) for trj_info in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).values()]
data = data[0]

# Get mask for observation region
node_coord = torch.tensor(data["pos"][0])
x_in_range = (node_coord[:, 0] >= vel_observe_region[0][0]) & (node_coord[:, 0] <= vel_observe_region[0][1])
y_in_range = (node_coord[:, 1] >= vel_observe_region[1][0]) & (node_coord[:, 1] <= vel_observe_region[1][1])
vel_observation_region_mask = x_in_range & y_in_range
vel_at_observation_true = torch.tensor(data["velocity"][inverse_timestep])[vel_observation_region_mask]

# Find true obstacle mask
ground_truth_node_type = torch.tensor(data["node_type"][0])
ground_truth_wall_mask = (ground_truth_node_type == NodeType.WALL_BOUNDARY)
x_in_range = (node_coord[:, 0] >= obstacle_region[0][0]) & (node_coord[:, 0] <= obstacle_region[0][1])
y_in_range = (node_coord[:, 1] >= obstacle_region[1][0]) & (node_coord[:, 1] <= obstacle_region[1][1])
obstacle_mask = x_in_range & y_in_range & torch.squeeze(ground_truth_wall_mask)

# # (Find center and radius of the true obstacle)
# obstacle_node_coords = node_coord[obstacle_mask_true]
# obstacle_center = [obstacle_node_coords[:, 0].mean(), obstacle_node_coords[:, 1].mean()]
# obstacle_radius = (obstacle_node_coords[:, 0].max() - obstacle_node_coords[:, 0].min())/2
# # See mesh nodes
# fig, ax = plt.subplots()
# ax.scatter(node_coord[:, 0], node_coord[:, 1], s=0.1)
# ax.scatter(obstacle_node_coords[:, 0], obstacle_node_coords[:, 1], color="red", s=0.1)
# plt.gca().set_aspect('equal')
# plt.show()


# Make a `node_type` without obstacle, which is the node type for the simulation domain
obstacle_index = torch.where(obstacle_mask)
nnode = ground_truth_node_type.shape
# Fill the node_type tensor to `NORMAL` if `obstacle_mask==True` else remain `ground_truth_node_type`
node_type_sim_domain = torch.where(
    obstacle_mask, torch.full(nnode, NodeType.NORMAL).squeeze(), ground_truth_node_type.squeeze()
).to(torch.float32)

# Save initial state as .npz which is the input for rollout
dim = torch.tensor(data["pos"][0].shape[1])
ncells = torch.tensor(data["cells"].shape[1])
nnode_per_cell = torch.tensor(data["cells"].shape[2])

# Initialize center and radius as tensors that require gradients
parameters = torch.tensor([0.2, 0.18, 0.1])  # center_x, center_y, radius

# Optimizer (simple stochastic gradient descent)
optimizer = torch.optim.Adam([parameters], lr=lr)

# Check for a saved state and load if it exists
output_path = f"{path}/outputs/"
optimization_state_file = "save.pt"
if not os.path.exists(output_path):
   os.makedirs(output_path)
if os.path.isfile(f"{output_path}/{optimization_state_file}") and resume:
    print("Resume from the previous state")
    checkpoint = torch.load(f"{output_path}/{optimization_state_file}")
    center = checkpoint['center']
    radius = checkpoint['radius']
    optimizer.load_state_dict(checkpoint['optimizer'])

# load simulator
simulator = learned_simulator.MeshSimulator(
    simulation_dimensions=2, nnode_in=11, nedge_in=3, latent_dim=128, nmessage_passing_steps=15, nmlp_layers=2,
    mlp_hidden_dim=128, nnode_types=3, node_type_embedding_size=9, device=device)
if os.path.exists(path + model_file):
    simulator.load(path + model_file)
else:
    raise Exception(f"Model does not exist at {path + model_file}")
simulator.to(device)
simulator.eval()

# Variables to save
history = []

# Training loop
for epoch in range(nepoch):
    optimizer.zero_grad()  # Clear previous gradients

    # get X_0: (cx, cy, r)
    current_node_type = utils.get_node_type(
            node_coord=node_coord,
            node_type_sim_domain=node_type_sim_domain,
            current_obstacle_location=parameters
        )
    initial_feature = (
            node_coord.reshape(1, nnode[0], dim),
            current_node_type.reshape(1, nnode[0], 1),
            torch.tensor(data["velocity"][0].reshape(1, nnode[0], dim)),  # TODO: velocity field at t=0 should correspond to the current obstacle location.
            torch.tensor(data["pressure"][0].reshape(1, nnode[0], 1)).detach(),
            torch.tensor(data["cells"][0].reshape(1, ncells, nnode_per_cell))
        )

    # get (X_0 + dh): [(cx+h1, cy, r), (cx, cy+h2, r), (cx+hc, cy, r+h3)]
    initial_feature_for_dh = []
    # Get initial state at each datapoint: (cx+h1, cy, r), (cx, cy+h2, r), (cx+hc, cy, r+h3)
    for dh in fd_perturb:
        # Get current node type considering current obstacle location
        current_node_type_dh = utils.get_node_type(
            node_coord=node_coord,
            node_type_sim_domain=node_type_sim_domain,
            current_obstacle_location=parameters + dh
        )

        # Make the feature of initial state for rollout
        initial_feature_dh = (
            node_coord.reshape(1, nnode[0], dim),
            current_node_type_dh.reshape(1, nnode[0], 1),
            torch.tensor(data["velocity"][0].reshape(1, nnode[0], dim)),  # TODO: velocity field at t=0 should correspond to the current+dh obstacle location.
            torch.tensor(data["pressure"][0].reshape(1, nnode[0], 1)).detach(),
            torch.tensor(data["cells"][0].reshape(1, ncells, nnode_per_cell))
        )
        # Append it for (cx+h1, cy, r), (cx, cy+h2, r), (cx+hc, cy, r+h3)
        initial_feature_for_dh.append(initial_feature_dh)

    # Rollout f(cx, cy, r) at t=inverse
    with torch.no_grad():
        velocity_pred = rollout(
            simulator=simulator,
            features=initial_feature,
            nsteps=inverse_timestep,
            device=device)
    vel_at_observation_pred = velocity_pred[-1][vel_observation_region_mask].detach().cpu()

    # Rollout for f(cx+h1, cy, r), f(cx, cy+h2, r), f(cx+hc, cy, r+h3)
    velocity_pred_for_dh = []
    vel_at_observation_pred_for_dh = []
    with torch.no_grad():
        for initial_feature_dh in initial_feature_for_dh:
            velocity_pred_dh = rollout(
                simulator=simulator,
                features=initial_feature_dh,
                nsteps=inverse_timestep,
                device=device)
            velocity_pred_for_dh.append(velocity_pred_dh)
            vel_at_observation_pred_for_dh.append(
                velocity_pred_dh[-1][vel_observation_region_mask].detach().cpu())

    # Compute the gradient for each variable using finite difference approximation
    # loss for each parameter perturb
    loss = torch.sqrt(
        torch.mean((vel_at_observation_pred - vel_at_observation_true)**2))
    loss_perturb = []
    for vel_at_observation_pred_dh in vel_at_observation_pred_for_dh:
        loss_dh = torch.sqrt(
            torch.mean((vel_at_observation_pred_dh - vel_at_observation_true)**2))
        loss_perturb.append(loss_dh)

    # Compute gradient of loss for each parameter
    grads = []
    for i, dh in enumerate(fd_perturb):
        grad = (loss_perturb[i] - loss) / dh[i]
        grads.append(grad)

    # Manually set the gradient
    parameters.grad = torch.tensor(grads)

    # Perform optimization step
    optimizer.step()

    # Print status
    if epoch % history_print_step == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.10f}")
        # print(f"Center: {parameters[0:1]}, radius: {parameters[2]}")

    # Save current rollout & true rollout
    results = {
        "pos": data["pos"],
        "ground_truth": {
            "node_type": data["node_type"][0],
            "vel": data["velocity"],
        },
        "prediction": {
            "node_type": current_node_type.detach().cpu().numpy(),
            "vel": velocity_pred.detach().cpu().numpy()
        }
    }
    with open(f'{output_path}/results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Save the optimization state at each iteration
    if epoch % save_step == 0:

        current_history = {
            "epcoh": epoch,
            "obstacle_parameters": parameters.detach().numpy(),
            "lr": optimizer.state_dict()["param_groups"][0]["lr"]
        }
        history.append(current_history)
        with open(f'{path}/history{epoch}.pkl', 'wb') as handle:
            pickle.dump(current_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save({
            'obstacle_parameters': parameters,
            'optimizer': optimizer.state_dict(),
        }, f"{output_path}/{optimization_state_file}")

    with open(f'{output_path}/histories.pkl', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot current optimization status
    utils.plot_flow(
        results=results, timestep=inverse_timestep, save_path=f"{path}/outputs/epoch{epoch}.png")


# %%
# true_center = torch.tensor([np.random.uniform(0.1, 0.6), np.random.uniform(0.05, 0.35)], requires_grad=True)
# true_radius = torch.tensor([np.random.uniform(0.08, 0.12)], requires_grad=True)
