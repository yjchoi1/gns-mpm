import pickle
from matplotlib import pyplot as plt
import os

data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/autograd_tall_phi20/record.pkl"

# open a file, where you stored the pickled data
file = open(data_path, 'rb')

# dump information to that file
record = pickle.load(file)

nepoch = len(record)
phi_guess = []
loss = []
for data in record:
    current_phi = data["current_phi"]
    updatad_phi = data["updated_phi"]
    phi_guess.append(updatad_phi)
    loss.append(data[f"loss"])


fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].scatter(range(nepoch), phi_guess)
ax[0].set_xlabel("Step")
ax[0].set_ylabel(r"$\phi$")
ax[0].set_ylim(bottom=0)
ax[1].scatter(range(nepoch), loss)
ax[1].set_xlabel("Step")
ax[1].set_ylim(bottom=0)
ax[1].set_ylabel("Loss")
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.show()

for data in record:
    updatad_phi = data["updated_phi"]
    target_position = data[f"rollout"]["target_final_positions"]
    predicted_position = data[f"rollout"]["predicted_final_positions"][:, -1, :]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.scatter(target_position[:, 0], target_position[:, 1], color="gray", alpha=0.5)
    ax.scatter(predicted_position[:, 0], predicted_position[:, 1], color="blue", alpha=0.3)
    ax.set_xlim([0.2, 0.35])
    ax.set_ylim([0, 0.5])
    plt.tight_layout()
    plt.show()



