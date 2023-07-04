import numpy as np
import pickle
import matplotlib as mpl
import sys
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/utils/')
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
import os
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from preprocess_rollout_data import get_positions, compute_kinemacis
from mpl_toolkits.axes_grid1 import ImageGrid


rollout_tags = \
    [
    # [("test2-1", "a=0.5", "solid", "0.9"),  # (simulation id, aspect ratio, linetype, linecolor)
    #  ("test4", "a=0.8", (0, (5, 1)), "0.7"),
    #  ("test0-2", "a=1.0", (0, (5, 5)), "0.5"),
    #  ("test5-1", "a=2.0", (0, (1, 1)), "0.3")]
    #  ("test5-5", "a=3.0", "dashdot", "0.15"),
    #  ("test6-1", "a=4.0", (0, (3, 5, 1, 5)), "0.0"),
     ("column_collapse9", "a=0.5", "dotted", "0.7")]

data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand3d-largesets-r041/"
save_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand3d-largesets-r041/"
target_rollout_filename = "rollout_column_collapse9_0_step8690000"
training_steps = 8690000
trajectory_ID = [0]
mpm_dt = 0.0025
gravity = 9.81
xboundary = [0.0, 1.5]
yboundary = [-0.25, 0.25]
zboundary = [0.0, 0.7]
output_normalized_timesteps = [0, 1.0, 2.5]
output_percentile = 100


rollout_filenames = []
for tag in rollout_tags:
    for i in trajectory_ID:
        if training_steps is not None:
            rollout_filename = f"rollout_{tag[0]}_{i}_step{training_steps}"
        else:
            rollout_filename = f"rollout_{tag[0]}_{i}"
        rollout_filenames.append(rollout_filename)


# get data
trajectories, metadata = get_positions(data_path, rollout_filenames, mpm_dt)
kinematics_info = compute_kinemacis(trajectories)
# Focus only one rollout as specified in the input
trajectory = trajectories[target_rollout_filename]
kinematics = kinematics_info[target_rollout_filename]

# get information for the runout data processing
timesteps = len(trajectory["gns"]["positions"])
normalized_time = trajectory["gns"]["normalized_times"]
L_initial = trajectory["gns"]["positions"][0, :, 0].max() - trajectory["gns"]["positions"][0, :, 0].min()
H_initial = trajectory["gns"]["positions"][0, :, 2].max() - trajectory["gns"]["positions"][0, :, 2].min()
g = 9.81 * mpm_dt ** 2  # because, in MPM, we used dt=0.0025, and, in GNS, dt=1.0
critical_time = np.sqrt(H_initial/g)
output_critical_timesteps = output_normalized_timesteps + [timesteps * mpm_dt/critical_time]  # append last timestep
output_timesteps = np.around(np.array(output_critical_timesteps) * critical_time, 0)
output_timesteps = list(output_timesteps.astype(int))
output_timesteps[-1] = timesteps - 1  # to comply with the last index of the list
# print(output_timesteps.astype(int))


# data to store
L_t = {}
H_t = {}
normalized_L_t = {}
normalized_H_t = {}
potential_E = {}
kinetic_E = {}
dissipation_E = {}
normalized_Ep = {}
normalized_Ek = {}
normalized_Ed = {}


# get data
for simulator, trj_data in trajectory.items():
    L_t[simulator] = []
    H_t[simulator] = []
    normalized_L_t[simulator] = []
    normalized_H_t[simulator] = []
    for position in trj_data["positions"]:
        H = position[:, 2].max() #- position[:, 2].min()  # top of runout
        L = np.percentile(position[:, 0], output_percentile) #- position[:, 0].min()   # front end of runout
        normalized_L = (L - L_initial) / L_initial
        normalized_H = H / L_initial
        L_t[simulator].append(L)
        H_t[simulator].append(H)
        normalized_L_t[simulator].append(normalized_L)
        normalized_H_t[simulator].append(normalized_H)

    mass = 1
    potential_E = np.sum(mass * g * trj_data["positions"][:, :, 2], axis=-1)  # sum(mass * gravity * elevation)
    kinetic_E = (1 / 2) * np.sum(mass * kinematics[simulator]['velocity'] ** 2, axis=-1)
    E0 = potential_E[0] + kinetic_E[0]
    dissipation_E = E0 - kinetic_E - potential_E
    # normalize energies
    normalized_Ek[simulator] = kinetic_E / E0
    normalized_Ep[simulator] = potential_E / E0
    normalized_Ed[simulator] = dissipation_E / E0



# plot for runout
# prepare plot
runout_fig, runout_ax = plt.subplots(figsize=(4.5, 3.2))
height_ax = runout_ax.twinx()  # make the figure have two y-axis
runout_legends = (
    ["MPM Runout", "GNS Runout"],
    ["MPM Height", "GNS Height"])
runout_lines = ["solid", "dashed"]
runout_colors = ["silver", "black"]
runout_p_sets = []

for i, sim_name in enumerate(["mpm", "gns"]):
    p1 = runout_ax.plot(normalized_time, normalized_L_t[sim_name],
                        color=runout_colors[i],
                        linestyle=runout_lines[0],
                        label=runout_legends[0][i])
    p2 = height_ax.plot(normalized_time, normalized_H_t[sim_name],
                        color=runout_colors[i],
                        linestyle=runout_lines[1],
                        label=runout_legends[1][i])
    # labels
    runout_ax.set_xlabel(r"$t / \tau_{c}$")
    runout_ax.set_ylabel(r"$(L_t - L_0)/L_0$")
    height_ax.set_ylabel(r"$H_t/L_0$")
    runout_ax.set_xlim(xmin=0, xmax=5)
    runout_ax.set_ylim(ymin=0, ymax=1.0)
    height_ax.set_xlim(xmin=0, xmax=5)
    height_ax.set_ylim(ymin=0, ymax=1.0)
    lines, labels = runout_ax.get_legend_handles_labels()
    lines2, labels2 = height_ax.get_legend_handles_labels()
    height_ax.legend(lines + lines2, labels + labels2,
                      ncol=2, loc="lower right", prop={'size': 8})

runout_fig.tight_layout()
plt.savefig(f"{save_path}/runout_{rollout_tags[0][0]}_step{training_steps}.png")
# plt.show()

# plot for energy
energy_fig, energy_ax = plt.subplots(figsize=(4.5, 3.2))
energy_ax2 = energy_ax.twinx()
energy_legends = (
    ["MPM $E_p/E_0$", "GNS $E_p/E_0$"],
    ["MPM $E_k/E_0$", "GNS $E_k/E_0$"],
    ["MPM $E_d/E_0$", "GNS $E_d/E_0$"])
energy_lines = ["solid", "dashed", "dotted"]
energy_colors = [["silver", "black"], ["lightcoral", "darkred"], ["lightsteelblue", "darkblue"]]
energy_p_sets = []

for i, sim_name in enumerate(["mpm", "gns"]):
    p3 = energy_ax.plot(normalized_time, normalized_Ep[sim_name],
                            color=energy_colors[0][i],
                            linestyle=energy_lines[0],
                            label=energy_legends[0][i])
    p4 = energy_ax2.plot(normalized_time, normalized_Ek[sim_name],
                                  color=energy_colors[1][i],
                                  linestyle=energy_lines[1],
                                  label=energy_legends[1][i])
    p5 = energy_ax2.plot(normalized_time, normalized_Ed[sim_name],
                                  color=energy_colors[2][i],
                                  linestyle=energy_lines[2],
                                  label=energy_legends[2][i])

    energy_ax.set_xlabel(r"$t / \tau_{c} $")
    energy_ax.set_ylabel(r"$E_p/E_0$")
    energy_ax.set_xlim(xmin=0, xmax=4)
    energy_ax.set_ylim(ymin=0, ymax=1.0)
    energy_p_set = p3 + p4 + p5
    energy_p_sets.extend(energy_p_set)
    energy_ax2.set_ylabel(r"$E_k/E_0$ and $E_d/E_0$")
    energy_ax2.set_ylim(ymin=0, ymax=0.5)
    # energy_ax2.set_ylim(ymin=0, ymax=0.3)
    # lines, labels = energy_ax.get_legend_handles_labels()
    # lines2, labels2 = energy_ax2.get_legend_handles_labels()
    # energy_ax2.legend(lines + lines2, labels + labels2,
    #                   ncol=3, loc=7, prop={'size': 8})
    # energy_ax.legend(ncol=2, loc="best", prop={'size': 8})
    # energy_ax.legend(ncol=2, loc="best", prop={'size': 8})
    # energy_ax2.legend(ncol=2, loc="best", prop={'size': 8})
    energy_fig.tight_layout()
# plt.show()
plt.savefig(f"{save_path}/energy_{rollout_tags[0][0]}_step{training_steps}.png")


for ((rollout_name, kin_data), (rollout_name, trj_data)) in zip(kinematics_info.items(), trajectories.items()):
    # get timesteps that corresponds to normalized times
    num_steps = len(trj_data["mpm"]["positions"])
    final_time = num_steps * mpm_dt
    L0 = trj_data["mpm"]["positions"][0][:, 0].max() - trj_data["mpm"]["positions"][0][:, 0].min()
    H0 = trj_data["mpm"]["positions"][0][:, 2].max() - trj_data["mpm"]["positions"][0][:, 2].min()
    critical_time = np.sqrt(H0 / 9.81)
    times_to_plot = np.array(output_normalized_timesteps) * critical_time
    timesteps_to_plot = np.round(times_to_plot / mpm_dt).astype(int)
    timesteps_to_plot = np.append(timesteps_to_plot, num_steps - 1)

    for i in timesteps_to_plot:

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(7.8, 2.8))
        # grid = ImageGrid(fig, 111,
        #                  nrows_ncols=(1, 2),
        #                  axes_pad=0.3,
        #                  cbar_location="right",
        #                  cbar_mode="single",
        #                  cbar_size="5%",
        #                  cbar_pad=0.05
        #                  )

        for j, (datacase, kinematics) in enumerate(kin_data.items()):
            # if j <= 1:
                # sample color values
                if datacase == "mpm":
                    cmap = plt.cm.viridis
                    vmax = np.ndarray.flatten(kinematics["displacement"]).max()
                    vmin = np.ndarray.flatten(kinematics["displacement"]).min()
                    sampled_value = kinematics["displacement"][i]

                # select ax to plot at set boundary
                # axs[j].set_aspect()
                axs[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axs[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                axs[j].set_zlim([float(zboundary[0]), float(zboundary[1])])
                trj = axs[j].scatter(trajectories[rollout_name][datacase]["positions"][i][:, 0],
                                      trajectories[rollout_name][datacase]["positions"][i][:, 1],
                                      trajectories[rollout_name][datacase]["positions"][i][:, 2],
                                      c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=2)
                axs[j].set_box_aspect(
                    aspect=(float(xboundary[1]) - float(xboundary[0]),
                            float(yboundary[1]) - float(yboundary[0]),
                            float(zboundary[1]) - float(zboundary[0])))
                # axes[j].grid(True, which='both')
                axs[j].set_title(datacase)
            # if j == 2:
                # fig.colorbar(trj, cax=axs.cbar_axes[0])
                fig.colorbar(trj, ax=axs[j], shrink=0.5, pad=0.15)

        plt.savefig(f"{save_path}/{rollout_name}_time{i}.png")
        plt.tight_layout()
        plt.show()

