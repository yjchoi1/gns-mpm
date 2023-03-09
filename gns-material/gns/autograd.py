import torch
import os
import sys
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
from gns import train


initial_conditions_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/sand2d_frictions338/initial_conditions.npz"
gns_metadata_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/sand2d_frictions338/"
nsteps = 380
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
noise_std = 6.7e-4
NUM_PARTICLE_TYPES = 9
model_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/sand2d_frictions-r015/"
model_file = "model-390000.pt"
target_runout = 1.0


def runout(
        simulator: learned_simulator.LearnedSimulator,
        initial_positions: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,  # torch scalar
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device):
    """
    Rolls out a trajectory by applying the model in sequence and -
    compute final runout distance of the column collapse at the edge
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
    material_property_expended = material_property * torch.full((1, n_particles_per_example), 1).to(device)

    for step in range(nsteps):
        print(f"rollout step {step}/{nsteps}")
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
            material_property=material_property_expended
        )

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    # Assuming that the column is on the left edge of 2d box domain, get lengths (L_0, L_f)
    initial_length = torch.max(initial_positions[:, 0, 0])
    final_length = torch.max(current_positions[:, -1, 0])
    normalized_runout = (final_length - initial_length / initial_length)

    return normalized_runout


# Load simulator
metadata = reading_utils.read_metadata(gns_metadata_path)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()  #??

# Load data containing MPM initial conditions (six initial position sequence, particle type, material feature)
dinit = data_loader.TrajectoriesDataset(path=initial_conditions_path)
# with torch.no_grad():
for example_i, features in enumerate(dinit):
    # Obtain features
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    positions = features[0].to(device)
    particle_type = features[1].to(device)
    material_property = features[2].to(device)
    n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)
    phi = material_property[0].clone().detach().requires_grad_(True)

    # Predict example rollout
    normalized_runout = runout(simulator, positions, particle_type, phi,
                               n_particles_per_example, nsteps, device)

    # Calculate the loss (i.e., mse between target runout and current runout with current phi)
    target = torch.tensor(target_runout)
    loss = torch.nn.functional.mse_loss(normalized_runout, target)
    print(f"Loss: {loss}")

    # Compute gradients of loss with respect to input material_type, phi
    loss.backward(retain_graph=True, inputs=[phi])

    # Access gradients of input material_type
    grads = phi.grad

    # Print the gradients
    print(f"Gradient of loss w.r.t phi is: {grads}")


