import torch
import sys
import os
import numpy as np

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
from gns import train


def forward_rollout(
        simulator: learned_simulator.LearnedSimulator,
        initial_positions: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,  # torch scalar
        n_particles_per_example: torch.tensor,
        nsteps: int,
        target,
        device):
    """
    Rolls out a trajectory by applying the model in sequence and -
    compute loss between final runout of the column collapse at the edge and target value
    Args:
        simulator: Learned simulator
        initial_positions: Torch tensor for initial position
        particle_types: Torch tensor for particle_types
        material_property: Torch tensor for particle_types, specifically `phi` in scalar
        n_particles_per_example: Torch tensor for n_particles_per_example
        nsteps: number of steps to predict
        device:

    Returns:
        runout distance
    """

    current_positions = initial_positions
    predictions = []
    material_property_expended = material_property * torch.full((1, n_particles_per_example), 1).to(device)

    print(f"Begin Rollout for {material_property.to(torch.float32)} rad...")
    for step in range(nsteps):
        if step % 100 == 0:
            print(f"rollout step {step}/{nsteps} at phi {material_property.to(torch.float32)} rad...")
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
            material_property=material_property_expended
        )
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    # Predictions with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    # loss
    loss = torch.mean((predictions[-1] - torch.from_numpy(target).to(device)) ** 2)

    output_dict = {
        'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'particle_types': particle_types.cpu().numpy(),
        'material_property': material_property.cpu().numpy(),
    }
    print(f"Completed rollout for {material_property.to(torch.float32)} rad...")

    return output_dict, loss.cpu().numpy()



