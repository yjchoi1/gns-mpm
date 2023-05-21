import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import os
import sys
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
from preprocess_rollout_data import get_positions, compute_kinemacis, kinematic_error


rollout_tags = \
[
    ("test2-1", "a=0.5", "solid", "0.9"),  # (simulation id, aspect ratio, linetype, linecolor)
    ("test4", "a=0.8", (0, (5, 1)), "0.7"),
    ("test0-2", "a=1.0", (0, (5, 5)), "0.5"),
    ("test5-1", "a=2.0", (0, (1, 1)), "0.3"),
    ("test5-5", "a=3.0", "dashdot", "0.15"),
    ("test6-1", "a=4.0", (0, (3, 5, 1, 5)), "0.0"),
    ("test4-2", "Upscaled a=0.8", "dotted", "0.7")]
data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand-small-r300-400step_serial/"
save_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand-small-r300-400step_serial/"
training_steps = 15270000
trajectory_ID = [0]
mpm_dt = 0.0025
gravity = 9.81

rollout_filenames = []
for tag in rollout_tags:
    for i in trajectory_ID:
        if training_steps is not None:
            rollout_filename = f"rollout_{tag[0]}_{i}_step{training_steps}"
        else:
            rollout_filename = f"rollout_{tag[0]}_{i}"
        rollout_filenames.append(rollout_filename)


# read positions for MPM and GNS from rollout.pkl
# get error
trajectories, metadata = get_positions(data_path, rollout_filenames, mpm_dt)
kinematics_info = compute_kinemacis(trajectories)
error_data, individual_error, mape = kinematic_error(kinematics_info)


# plot errors for each rollout
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for i, (rollout_name, data) in enumerate(mape.items()):
    for j, (kinematic_type, value) in enumerate(data.items()):
        normalized_times = trajectories[rollout_name]["mpm"]["normalized_times"]
        ax[j].plot(normalized_times, value, linestyle=rollout_tags[i][2], color=rollout_tags[i][3], label=rollout_tags[i][1])
        ax[j].set_xlabel(r"$t / \tau_c$")
        ax[j].set_ylabel(f"MAPE for {kinematic_type}")
        ax[j].set_xlim(left=0, right=8)
        # ax[j].set_ylim(bottom=0, top=3e-3)
        ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # ax[j].set_yscale("log")
        ax[j].legend(loc='best', prop={'size': 9})
    # fig.show()
    plt.tight_layout()
    fig.savefig(f"{save_path}/error.png")

# plot errors for individual particles for each rollout
for i, (rollout_name, data) in enumerate(individual_error.items()):
    fig, ax = plt.subplots(2, 3, figsize=(18, 7))
    for j, (kinematic_type, value) in enumerate(data.items()):
        normalized_times = trajectories[rollout_name]["mpm"]["normalized_times"]
        final_disps = kinematics_info[rollout_name]["gns"]["displacement"][-1, :]
        max_for_time = np.max(value, axis=1)
        min_for_time = np.min(value, axis=1)
        std_for_time = np.std(value, axis=1)
        mean_for_time = np.mean(value, axis=1)
        median_for_time = np.median(value, axis=1)

        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=final_disps.min(), vmax=final_disps.max())
        for c, nth_particle in enumerate(range(value.shape[1])):
            indiv_err = ax[0, j].plot(normalized_times, value[:, nth_particle],
                                      alpha=0.5, color=cmap(norm(final_disps[c])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # ax[0, j].plot(normalized_times, median_for_time,
        #               label="Median", color="red", linestyle="dashed")
        # ax[0, j].plot(normalized_times, mean_for_time + std_for_time,
        #               label=r"$+ \sigma$", color="red", linestyle="dotted")
        # ax[0, j].plot(normalized_times, mean_for_time - std_for_time,
        #               label=r"$- \sigma$", color="red", linestyle="dotted")
        ax[0, j].set_xlabel(r"$t / \tau_c$")
        ax[0, j].set_ylabel(f"Error for {kinematic_type}")
        # ax[0, j].legend(loc='best')
        ax[0, j].set_xlim(left=0, right=6.2)
        # ax[0, j].set_yscale("log")
        ax[0, j].set_ylim(bottom=0)
        ax[0, j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax[1, j].plot(normalized_times, mean_for_time, label='Mean')
        ax[1, j].fill_between(
            x=normalized_times,
            y1=min_for_time,
            y2=max_for_time,
            alpha=0.2,
            label='Min & max band')
        ax[1, j].set_xlabel(r"$t / \tau_c$")
        ax[1, j].set_ylabel(f"MSE for {kinematic_type}")
        ax[1, j].set_yscale("log")
        # plt.legend()

    fig.suptitle(f'{rollout_tags[i][1]}', fontsize=10)
    cbar = plt.colorbar(sm, ax=ax[0, 0])
    cbar.set_label('Final displacement')
    cbar.vmin = 0

    # fig.show()
    plt.tight_layout()
    fig.savefig(f"{save_path}/error_indiv_{rollout_tags[i][1]}.png")

# # plot spacial distribution of error evolution
# for i, (rollout_name, data) in enumerate(individual_error.items()):
#     positions = trajectories[rollout_name]["gns"]["positions"]
#     normalized_times = trajectories[rollout_name]["mpm"]["normalized_times"]
#     time_samples = np.linspace(0, len(normalized_times), 10, endpoint=False).astype(int)
#     for t in time_samples:
#         fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
#         for j, (kinematic_type, value) in enumerate(data.items()):
#             vmax = np.ndarray.flatten(value).max()
#             vmin = np.ndarray.flatten(value).min()
#             ax = axs[j]
#             sampled_value = value[t]
#             geometry = ax.scatter(positions[t, :, 0], positions[t, :, 1],
#                                   c=sampled_value, vmin=vmin, vmax=vmax)
#             ax.set_xlim(metadata["bounds"][0])
#             ax.set_ylim(metadata["bounds"][1])
#             ax.set_aspect('equal')
#             ax.set_title(f"MSE for {kinematic_type}")
#             fig.colorbar(geometry, ax=ax, shrink=0.5)
#         plt.tight_layout()
#         sampled_normalized_times = normalized_times[t]
#         plt.suptitle(f"{rollout_tags[i][1]} at {sampled_normalized_times:.2f}")
#         plt.savefig(f"{save_path}/error_geom_{rollout_tags[i][1]}_at{sampled_normalized_times}.png")
#         # plt.show()
# a = 1
#
#
#
#
