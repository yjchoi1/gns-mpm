import matplotlib.pyplot as plt
import torch
import json
import numpy as np
import itertools
import sys
import pickle

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns/')
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import train

data_name = "sand2d_frictions-r015"
data_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/{data_name}/"
model_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/{data_name}/"
output_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/{data_name}/"
train_history_loc = f"{model_path}/loss_hist.pkl"

batch_size = 2
INPUT_SEQUENCE_LENGTH = 6
noise_std = 6.7e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
n_features = 3
eval_steps = np.arange(10000, 100000, 10000)
nexamples_for_loss = 20


def get_loss_history(eval_steps, dataset, nexamples_for_loss):
    # load simulator
    metadata = reading_utils.read_metadata(data_path)
    simulator = train._get_simulator(metadata, noise_std, noise_std, device)

    with torch.no_grad():
        loss_per_step = []

        for step in eval_steps:
            print(f"Load simulator at step {step}")
            simulator.load(model_path + f"model-{step}.pt")
            simulator.to(device)
            simulator.eval()
            total_loss = []

            for i, example in itertools.islice(enumerate(dataset),
                                               nexamples_for_loss):  # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
                print(i)

                position = example[0][0]
                particle_type = example[0][1]
                if n_features == 3:  # if dl includes material_property
                    material_property = example[0][2]
                    n_particles_per_example = example[0][3]
                elif n_features == 2:
                    n_particles_per_example = example[0][2]
                else:
                    raise NotImplementedError
                labels = example[1]

                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position, noise_std_last_step=noise_std).to(device)
                non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
                sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                pred_acc, target_acc = simulator.predict_accelerations(
                    next_positions=labels.to(device),
                    position_sequence_noise=sampled_noise.to(device),
                    position_sequence=position.to(device),
                    nparticles_per_example=n_particles_per_example.to(device),
                    particle_types=particle_type.to(device),
                    material_property=material_property.to(device) if n_features == 3 else None
                )

                loss = (pred_acc - target_acc) ** 2
                loss = loss.sum(dim=-1)
                num_non_kinematic = non_kinematic_mask.sum()
                loss = torch.where(non_kinematic_mask.bool(),
                                   loss, torch.zeros_like(loss))
                loss = loss.sum() / num_non_kinematic
                loss = loss.to("cpu").numpy()
                print(f"Loss {i}/{nexamples_for_loss}: {loss} at model-{step}")
                total_loss.append(loss)

            mean_loss = np.mean(total_loss)
            loss_per_step.append(mean_loss)

    loss_history = np.vstack((eval_steps, loss_per_step))
    loss_history = loss_history.transpose()

    return loss_history


ds = data_loader.get_data_loader_by_samples(
    path=f"{data_path}/train.npz",
    input_length_sequence=INPUT_SEQUENCE_LENGTH,
    batch_size=batch_size)

val_history = get_loss_history(
    eval_steps=eval_steps,
    dataset=ds,
    nexamples_for_loss=nexamples_for_loss)

# Save loss history
save_name = f"val_history"
with open(f"{output_path}/{save_name}.pkl", 'wb') as f:
    pickle.dump(np.transpose(val_history), f)

# Load train history
with open(train_history_loc, 'rb') as f:
    train_history_data = pickle.load(f)
# convert torch tensor to numpy array
data_nps = []
for j in range(len(train_history_data)):
    data_np = train_history_data[j][0], train_history_data[j][1].detach().to(
        'cpu').numpy()  # shape=(nsave_steps, 2=(step, loss))
    data_nps.append(data_np)
train_loss_hist = np.array(data_nps)

# plot
fig, ax = plt.subplots()
ax.plot(train_loss_hist[:, 0], train_loss_hist[:, 1], lw=1, alpha=0.5, label="train")
ax.plot(val_history[:, 0], val_history[:, 1], alpha=0.5, label="validation", color="red")
ax.set_yscale('log')
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_xlim([0, 40000])
# ax.set_ylim([10e-5, 2])
plt.show()
