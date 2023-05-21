import matplotlib.pyplot as plt
import torch
import json
import numpy as np
import itertools
import sys
import pickle
# import seaborn as sns
import pandas as pd
from torch.utils.data import Subset

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns/')
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import train

data_name = "sand3d-largesets-r041"
data_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/{data_name}/"
model_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/{data_name}/"
output_path = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/models/{data_name}/"
valid_metadata = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/{data_name}/valid/"
train_metadata = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/{data_name}/train/"
# train_history_loc = f"{model_path}/loss_hist.pkl"

batch_size = 1
INPUT_SEQUENCE_LENGTH = 6
noise_std = 6.7e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
n_features = 2
# eval_steps = np.append(np.arange(0, 1000000, 20000), np.arange(1000000, 15000000, 100000))
eval_steps = np.arange(0, 8500000, 20000)
nexamples_for_loss = 5


def get_loss_history(eval_steps, dataset, nexamples_for_loss, metadata_path):
    # load simulator
    metadata = reading_utils.read_metadata(metadata_path)
    simulator = train._get_simulator(metadata, noise_std, noise_std, device)

    with torch.no_grad():
        loss_per_step = []

        for step in eval_steps:
            print(f"Load simulator at step {step}")
            simulator.load(model_path + f"model-{step}.pt")
            simulator.to(device)
            simulator.eval()
            total_loss = []

            # # Subsample dataset
            # len_dataset = len(dataset)
            # indices = np.arange(len_dataset)
            # indices = np.random.permutation(indices)
            # sampling_indices = indices[:nexamples_for_loss]
            # dataset = Subset(ds_train, sampling_indices)

            for i, example in itertools.islice(enumerate(dataset),
                                               nexamples_for_loss):  # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
                # print(i)

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
            print(f"Mean loss at {step}: {mean_loss}")
            loss_per_step.append(mean_loss)

    loss_history = np.vstack((eval_steps, loss_per_step))
    # loss_history = loss_history.transpose()

    return loss_history

#
ds_train = data_loader.get_data_loader_by_samples(
    path=f"{data_path}/train.npz",
    input_length_sequence=INPUT_SEQUENCE_LENGTH,
    batch_size=batch_size)

ds_val = data_loader.get_data_loader_by_samples(
    path=f"{data_path}/valid.npz",
    input_length_sequence=INPUT_SEQUENCE_LENGTH,
    batch_size=batch_size)

train_history = get_loss_history(
    eval_steps=eval_steps,
    dataset=ds_train,
    nexamples_for_loss=nexamples_for_loss,
    metadata_path=train_metadata)

valid_history = get_loss_history(
    eval_steps=eval_steps,
    dataset=ds_val,
    nexamples_for_loss=nexamples_for_loss,
    metadata_path=valid_metadata)

# Save loss history
save_name = f"train_history"
with open(f"{output_path}/{save_name}.pkl", 'wb') as f:
    pickle.dump(np.transpose(train_history), f)
save_name = f"valid_history"
with open(f"{output_path}/{save_name}.pkl", 'wb') as f:
    pickle.dump(np.transpose(valid_history), f)

# # Load original train history
# with open(train_history_loc, 'rb') as f:
#     whole_train_history = pickle.load(f)
# # convert torch tensor to numpy array
# data_nps = []
# for j in range(len(whole_train_history)):
#     data_np = whole_train_history[j][0], whole_train_history[j][1].detach().to(
#         'cpu').numpy()  # shape=(nsave_steps, 2=(step, loss))
#     data_nps.append(data_np)
# train_loss_hist = np.array(data_nps)

# Load computed loss histories
save_name = f"train_history"
with open(f"{output_path}/{save_name}.pkl", 'rb') as f:
    train_history_data = pickle.load(f)
save_name = f"valid_history"
with open(f"{output_path}/{save_name}.pkl", 'rb') as f:
    valid_history_data = pickle.load(f)

# plot
val_df = pd.DataFrame(valid_history_data, columns=["step", "loss"])
train_df = pd.DataFrame(train_history_data, columns=["step", "loss"])
# train_df = train_df.sort_values("step")

# Calculate moving average
sample_interval = 1
window_val = 10
window_train = 5
val_rolling_mean = val_df["loss"].rolling(window=window_val, center=True).mean()
train_rolling_mean = train_df["loss"].rolling(window=window_train, center=True).mean()

fig, ax = plt.subplots(figsize=(5, 3))
# ax.plot(train_history_data[:, 0], train_history_data[:, 1], lw=1, alpha=0.5, label="train")
# ax.plot(valid_history_data[:, 0], valid_history_data[:, 1], alpha=0.5, label="Validation")
ax.plot(valid_history_data[:, 0], val_rolling_mean, alpha=0.5, label="validation")
ax.plot(train_history_data[:, 0], train_rolling_mean, alpha=0.5, label="Training")
ax.set_yscale('log')
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_xlim([0, 15000000])
ax.legend()
plt.tight_layout()
plt.savefig('loss_hist.png')
plt.show()



# fig, ax = plt.subplots()
# # ax.plot(train_history_data[:, 0], train_history_data[:, 1], lw=1, alpha=0.5, label="train")
# ax.scatter(valid_history_data[:, 0], valid_history_data[:, 1], alpha=0.5, label="validation")
# ax.scatter(train_loss_hist[::10, 0], train_loss_hist[::10, 1], alpha=0.5, label="train_original")
# ax.set_yscale('log')
# ax.set_xlabel("Step")
# ax.set_ylabel("Loss")
# ax.set_xlim([0, 15000000])
# # ax.set_ylim([10e-5, 2])
# ax.legend()
# plt.show()
# plt.savefig('loss_hist.png')
