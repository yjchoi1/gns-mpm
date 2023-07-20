import torch
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import sys
import torch_geometric.transforms as T

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/')
from meshnet import learned_simulator
from meshnet.utils import NodeType
from meshnet.utils import datas_to_graph
from meshnet.transform_4face import MyFaceToEdge


INPUT_SEQUENCE_LENGTH = 1
transformer = T.Compose([MyFaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
dt = 0.001


def rollout(simulator: learned_simulator.MeshSimulator,
            features,
            nsteps: int,
            device):

    node_coords = features[0]  # (1, nnode, ndims)
    node_types = features[1]  # (1, nnode, 1)
    velocities = features[2]  # (1, nnode, ndims)
    pressures = features[3]  # (1, nnode, 1)
    cells = features[4]  # # (1, ncells, nnode_per_cell)

    initial_velocities = velocities[:INPUT_SEQUENCE_LENGTH]
    ground_truth_initial_velocity = velocities[:INPUT_SEQUENCE_LENGTH].squeeze().to(device)

    current_velocities = initial_velocities.squeeze().to(device)
    predictions = []
    mask = None

    for step in tqdm(range(nsteps), total=nsteps):

        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords = node_coords[0]
        current_node_type = node_types[0]
        current_pressure = pressures[0]
        current_cell = cells[0]
        current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        current_example = (
            (current_node_coords, current_node_type, current_velocities, current_pressure, current_cell, current_time_idx_vector),
            ground_truth_initial_velocity)

        # Make graph
        graph = datas_to_graph(current_example, dt=dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph)

        # Predict next velocity
        predicted_next_velocity = simulator.predict_velocity(
            current_velocities=graph.x[:, 1:3],
            node_type=graph.x[:, 0],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr)

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the same for the later timesteps
            mask = torch.logical_or(current_node_type == NodeType.NORMAL, current_node_type == NodeType.OUTFLOW)
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1)
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        predicted_next_velocity[mask] = ground_truth_initial_velocity[mask]
        predictions.append(predicted_next_velocity)

        # Update current position for the next prediction
        current_velocities = predicted_next_velocity.to(device)

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    return predictions
