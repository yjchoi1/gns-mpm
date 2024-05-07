# yc: Result analysis for inverse analysis

import torch
import os
import glob
import re
import numpy as np
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import pyplot as plt

read_idx = range(1, 11)
path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/"
sim_names = ["short_phi21", "short_phi42", "tall_phi21", "tall_phi42"]
output_case = "outputs_ad_const_lim0.0005_mag4000_lr500"
true_phi = [21, 42]
initial_phi = 30

# Init data storage
result_dict = {}
for sim_name in sim_names:
    result_dict[sim_name] = {
        "epochs": [],
        "loss_hist": [],
        "friction_hist": [],
        "final_deposit_true": [],
        "final_deposit_pred": []
    }

# Sort the filenames based on the trailing digits
def extract_number(filename):
    match = re.search(r'optimizer_state-(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # default if no match

# Get data
for sim_name in sim_names:
    record_dir = os.path.join(path, sim_name, output_case, 'optimizer_state-*')
    record_files = glob.glob(record_dir)
    record_files = sorted(record_files, key=extract_number)

    for i, record_file in enumerate(record_files):
        nepochs = len(record_files)
        record = torch.load(record_file)

        # pick data to store
        epoch = record['epoch']
        loss = record['loss'].item()
        friction = record['friction_state_dict']['current_params'].cpu().numpy()
        # Store
        result_dict[sim_name]["epochs"].append(epoch)
        result_dict[sim_name]["loss_hist"].append(loss)
        result_dict[sim_name]["friction_hist"].append(friction)

        # pick the position of the last deposit
        ground_truth_positions = record['position_state_dict']['target_positions']
        predicted_positions = record['position_state_dict']['inversion_positions']
        # Store
        # Note that the original data has an error that `true` and `pred` is changed.
        result_dict[sim_name]["final_deposit_pred"].append(predicted_positions[-1])
        result_dict[sim_name]["final_deposit_true"].append(ground_truth_positions[-1])

# Plot friction hist
sim_name_labels = [r"Short $\phi_{target}=21 \degree$",
                   r"Short $\phi_{target}=42 \degree$",
                   r"Tall $\phi_{target}=21 \degree$",
                   r"Tall $\phi_{target}=42 \degree$"]
colors = ["#FFA500", "#FFA500", "#00BFFF", "#00BFFF"]
markers = ["o", "s", "o", "s"]
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, sim_name in enumerate(sim_names):
    epochs = [0] + result_dict[sim_name]["epochs"]
    friction_hist = [initial_phi] + result_dict[sim_name]["friction_hist"]
    ax.plot(epochs, friction_hist,
            c=colors[i], marker=markers[i], label=sim_name_labels[i])
ax.set_xlim([0, 40])
ax.set_ylim([17, 48])
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Friction angle, $\phi$ ($\degree$)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{path}/phi_hist.png")

# Plot final deposit hist
case = "tall_phi21"
iteration_to_plot = [0, 2, 5, 10, 13, 25, 35, 38]
pred_frictions = [30.0] + result_dict[case]['friction_hist']

for i, (deposit_pred, deposit_true) in enumerate(
        zip(result_dict[case]["final_deposit_pred"],
            result_dict[case]["final_deposit_true"])
):
    if i in iteration_to_plot:
        # Process data to get perimeter of true runout
        runout_perimeter = alphashape.alphashape(
            deposit_true[:, :], alpha=10.0)  # smaller alpha fit more tight

        fig, ax = plt.subplots(figsize=(4, 2.5))
        # ax.scatter(deposit_true[:, 0], deposit_true[:, 1],
        #            c="yellow", edgecolor='k', alpha=0.0, s=20.0, label="True", zorder=10)
        ax.scatter(deposit_pred[:, 0], deposit_pred[:, 1],
                   c="gold", s=20.0, label="Prediction")

        if isinstance(runout_perimeter, Polygon):
            x, y = runout_perimeter.exterior.xy
            ax.fill(x, y, alpha=0.2,
                    color='black',
                    linewidth=3,
                    label='true')
        elif isinstance(runout_perimeter, MultiPolygon):
            for geom in runout_perimeter.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.2,
                        color='black',
                        linewidth=3,
                        label='True')

        ax.set_xlim([0, 0.9])
        ax.set_ylim([0, 0.43])
        ax.set_title(rf"Iteration {i}, $\phi$={float(pred_frictions[i]):.2f} $\degree$")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/{case}/final_deposit-{i}.png")

        plt.show()

a=1