import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle
from forward import rollout

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/inverse-flow')
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/meshnet/')
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
nepoch = 10
radius_tol = 0.00001
# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
history_print_step = 1
save_step = 1


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
center = torch.tensor([0.2, 0.18], requires_grad=True)
radius = torch.tensor([0.1], requires_grad=True)

# Optimizer (simple stochastic gradient descent)
optimizer = torch.optim.SGD([center, radius], lr=0.001)

# Check for a saved state and load if it exists
save_file = f"{path}/save.pth"
if os.path.isfile(save_file) and resume:
    print("Resume from the previous state")
    checkpoint = torch.load(save_file)
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

    # Create `current_node_type` tensor considering the current center and radius of circular obstacle
    # First,get `current_obstacle_mask`: true if `node_coord` is in the circular obstacle
    distances = torch.sqrt(torch.sum((node_coord - center) ** 2, dim=1))
    current_obstacle_mask = (distances <= radius + radius_tol)
    # Next, make `current_node_type` tensor:
    # fill with `WALL_BOUNDARY` if `current_obstacle_mask`, else, maintain `node_type_sim_domain`.
    current_node_type = torch.where(
        current_obstacle_mask,
        torch.full(nnode, NodeType.WALL_BOUNDARY).squeeze(),
        node_type_sim_domain.squeeze()
    ).to(torch.float32)

    # Make the feature of initial state for rollout
    initial_feature = (
        node_coord.reshape(1, nnode[0], dim),
        current_node_type.reshape(1, nnode[0], 1),
        torch.from_numpy(data["velocity"][0].reshape(1, nnode[0], dim)),
        torch.from_numpy(data["pressure"][0].reshape(1, nnode[0], 1)).detach(),
        torch.from_numpy(data["cells"][0].reshape(1, ncells, nnode_per_cell))
    )

    # Rollout
    velocity_pred = rollout(
        simulator=simulator,
        features=initial_feature,
        nsteps=inverse_timestep,
        device=device)

    # save current rollout & true rollout
    results = {
        "pos": data["pos"],
        "ground_truth": {
            "node_type": data["node_type"],
            "vel": data["velocity"],
        },
        "prediction": {
            "node_type": current_node_type.detach().cpu().numpy(),
            "vel": velocity_pred.detach().cpu().numpy()
        }
    }
    with open(f'{path}/results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # RMSE
    vel_pred_at_observation_pred = velocity_pred[-1][vel_observation_region_mask].to(device)
    vel_at_observation_true = vel_at_observation_true.to(device)
    loss = torch.sqrt(torch.mean((vel_pred_at_observation_pred - vel_at_observation_true)**2))

    # Compute gradients and update initial velocities
    loss.backward()
    print(center.grad)
    print(radius.grad)

    optimizer.step()

    # Save the optimization state at each iteration
    if epoch % save_step == 0:
        current_history = {}
        current_history["epcoh"] = epoch
        current_history["centers"] = center.detach().numpy()
        current_history["radius"] = radius.detach().item()
        current_history["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
        history.append(current_history)
        torch.save({
            'center': center,
            'radius': radius,
            'optimizer': optimizer.state_dict(),
        }, save_file)

    if epoch % history_print_step == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.10f}")
        print(f"Center: {center.detach()}, radius: {radius.detach()}")


# %%
# true_center = torch.tensor([np.random.uniform(0.1, 0.6), np.random.uniform(0.05, 0.35)], requires_grad=True)
# true_radius = torch.tensor([np.random.uniform(0.08, 0.12)], requires_grad=True)
