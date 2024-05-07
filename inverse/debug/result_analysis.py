import torch
import numpy as np
from matplotlib import pyplot as plt

read_idx = range(1, 11)
path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi42/outputs_fd_const_lim0.0005_mag1_lr1000/"
true_phi = 21
initial_phi = 30

# Get data
epochs = []
loss_hist = []
friction_hist = []
for i in read_idx:
    record = torch.load(f"{path}/optimizer_state-{i}.pt")
    epoch = record['epoch']
    loss = record['loss'].item()
    friction = record['friction_state_dict']['current_params'].cpu().numpy()
    if i == read_idx[-1]:
        ground_truth_positions = record['position_state_dict']['target_positions']
        predicted_positions = record['position_state_dict']['inversion_positions']

    epochs.append(epoch)
    loss_hist.append(loss)
    friction_hist.append(friction)

# Plot friction hist
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(epochs, friction_hist,
        c="black", alpha=0.7, linewidth=2.0, label="True")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Friction angle (\degree)")
plt.tight_layout()
plt.savefig(f"{path}/phi_hist.png")

# Plot loss
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(epochs, loss_hist)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss (MSE)")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{path}/loss_hist.png")
