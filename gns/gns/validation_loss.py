import pickle
import numpy as np
import torch
import os
# os.chdir("/work2/08264/baagee/frontera/gns-mpm/gns")
from matplotlib import pyplot as plt
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import train



# Inputs
data_path = "../gns-data/datasets/sand-small-r300_parallel_dwnsmp18/"
model_path = "../gns-data/models/sand-small-r300_parallel_dwnsmp18/"
output_path = "../gns-data/models/sand-small-r300_parallel_dwnsmp18/"
data_name = 'train-small-400step-val'
metadata = reading_utils.read_metadata(data_path)
noise_std = 6.7e-4
INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
batch_size = 2
steps = np.arange(0, 4000000, 20000)  # Training steps to evaluate loss
ntrajectory = 1  # The number of trajectories to evaluate loss


# Set device and import simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = train._get_simulator(metadata, noise_std, noise_std, device)

# load dataset
ds = data_loader.get_data_loader_by_samples(
    path=f"{data_path}{data_name}.npz",
    input_length_sequence=INPUT_SEQUENCE_LENGTH,
    batch_size=batch_size)

def onestep_loss(model_file, ds, eval_points):

    # load simulator
    simulator.load(model_path + model_file)
    simulator.to(device)

    eval_loss = []
    # evaluate mean one-step loss for "eval_points" number of examples
    for i, ((position, particle_type, n_particles_per_example), labels) in enumerate(ds):
        if i in eval_points:
            position.to(device)
            particle_type.to(device)
            n_particles_per_example = n_particles_per_example.to(device)
            labels.to(device)

            # Sample the noise to add to the inputs to the model during training.
            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                position, noise_std_last_step=noise_std).to(device)
            non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
            sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

            # Get the predictions and target accelerations.
            pred_acc, target_acc = simulator.predict_accelerations(
                next_positions=labels.to(device),
                position_sequence_noise=sampled_noise.to(device),
                position_sequence=position.to(device),
                nparticles_per_example=n_particles_per_example.to(device),
                particle_types=particle_type.to(device))

            # Calculate the loss and mask out loss on kinematic particles
            loss = (pred_acc - target_acc) ** 2
            loss = loss.sum(dim=-1)
            num_non_kinematic = non_kinematic_mask.sum()
            loss = torch.where(non_kinematic_mask.bool(),
                               loss, torch.zeros_like(loss))
            loss = loss.sum() / num_non_kinematic
            # print(f"At {model_file} dataset {i}: loss={loss}")
            eval_loss.append(float(loss))
    mean_loss = sum(eval_loss)/len(eval_loss)
    print(f"Mean loss at {len(eval_points)} for {model_file}: {mean_loss}")
    return mean_loss

# Model files to evaluate loss
model_files = []
for step in steps:
    model_file = f"model-{step}.pt"
    model_files.append(model_file)

# Evaluate loss at the specified steps
loss_history = []
eval_points = np.random.uniform(0, len(ds), ntrajectory).astype(int)
for model_file in model_files:
    eval_loss = onestep_loss(model_file, ds, eval_points)
    loss_history.append(eval_loss)
loss_history = np.vstack((steps, loss_history))

# Save loss history
save_name = f"loss_onestep_{data_name}"
with open(f"{output_path}{save_name}.pkl", 'wb') as f:
    pickle.dump(np.transpose(loss_history), f)

# fig, ax = plt.subplots()
# ax.plot(steps, eval_losses)
# ax.set_yscale('log')
# fig.show()
# fig.savefig("loss_mean.png")