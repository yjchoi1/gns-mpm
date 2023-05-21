import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import os
import sys
import matplotlib.colors as colors
import argparse

if __name__ == "__main__":

    # #%% Inputs for mesh, node, node_sets
    # parser = argparse.ArgumentParser(description='Make rollout analysis plots')
    # parser.add_argument('--rollout_path', help="Path to rollout")
    # parser.add_argument('--rollout_filename', help='File name of the rollout to analyse')
    # parser.add_argument('--output_percentile', default=100, type=float, help="Percentile from the farthest runout where the analysis based on")
    # parser.add_argument('--output_filename', help='Output file name to save plots')
    # args = parser.parse_args()

    rollout_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand2d_frictions-sr020/"
    rollout_filename = "rollout_eval15_0_step6300000"
    output_percentile = 99.9
    output_filename = "step6300000_eval15"
    mass = 1
    # rollout_path = args.rollout_path
    # rollout_filename = args.rollout_filename
    # output_percentile = args.output_percentile
    # output_filename = args.output_filename
    # mass = 1

    # # %% inputs for read rollout.pkl
    # rollout_path = "../gns-data/rollouts/sand-small-r190-bound"
        # rollout_filename = "rollout_test0-2_0"


    # %% Read rollout data
    with open(os.path.join(rollout_path, rollout_filename + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)

    # get trajectories
    predicted_trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data["predicted_rollout"]], axis=0)
    mpm_trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data["ground_truth_rollout"]], axis=0)
    trajectories = (mpm_trajectory, predicted_trajectory)
    height = np.max(trajectories[0][0][:, 1])

    # get information for the runout
    boundary = rollout_data["metadata"]["bounds"]
    timesteps = rollout_data["metadata"]["sequence_length"]
    dt = rollout_data["metadata"]["dt"]
    g = 9.81 * dt ** 2  # because, in MPM, we used dt=0.0025, and, in GNS, dt=1.0
    critical_time = np.sqrt(height/g)
    output_critical_timesteps = [0, 2.5, 4.0, timesteps/critical_time]
    output_timesteps = np.around(np.array(output_critical_timesteps) * critical_time, 0)
    output_timesteps = list(output_timesteps.astype(int))
    output_timesteps[-1] = timesteps - 1  # to comply with the last index of the list
    # print(output_timesteps.astype(int))
    # sys.exit()

    # %% Prepare plots
    # runout and height
    runout_fig, runout_ax = plt.subplots(figsize=(4.5, 3.2))
    height_ax = runout_ax.twinx()  # make the figure have two y-axis
    runout_legends = (
        ["MPM Runout", "GNS Runout"],
        ["MPM Height", "GNS Height"])
    runout_lines = ["solid", "dashed"]
    runout_colors = ["silver", "black"]
    runout_p_sets = []

    # energy
    energy_fig, energy_ax = plt.subplots(figsize=(4.5, 3.2))
    energy_ax2 = energy_ax.twinx()
    energy_legends = (
        ["MPM $E_p/E_0$", "GNS $E_p/E_0$"],
        ["MPM $E_k/E_0$", "GNS $E_k/E_0$"],
        ["MPM $E_d/E_0$", "GNS $E_d/E_0$"])
    energy_lines = ["solid", "dashed", "dotted"]
    energy_colors = [["silver", "black"], ["lightcoral", "darkred"], ["lightsteelblue", "darkblue"]]
    energy_p_sets = []

    # # color values for velocity contour
    # color_values = []
    # for i, trajectory in enumerate(trajectories):
    #     ## Extract data from runout
    #     # velocities
    #     velocities = trajectory[1:, ] - trajectory[:-1, ]  # velocities for x and y
    #     initial_velocity = velocities[0]  # copy initial velocity
    #     initial_velocity = np.expand_dims(initial_velocity, 0)  # add third dimension for concat later with velocities
    #     velocities = np.concatenate((velocities, initial_velocity))  # Concatenate
    #     scalar_velocities = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2)  # scalar sum of x y velocities
    #     color_values.append(scalar_velocities)
    #     # runout height (H) and length (L)
    # color_values = np.concatenate(color_values)
    # flat_color_values = np.ndarray.flatten(color_values)
    # max_vel = np.amax(flat_color_values)
    # min_vel = np.amin(flat_color_values)
    # plot_color = np.linspace(min_vel, max_vel, np.shape(color_values)[1])

    # color values for displacement
    whole_disps = []
    for i, trajectory in enumerate(trajectories):
        initial_position = trajectory[0, :]
        for current_position in trajectory:
            displacement = initial_position - current_position
            scalr_disp = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
            whole_disps.append(scalr_disp)
    whole_disps = np.concatenate(whole_disps)
    max_disp = np.amax(whole_disps)
    min_disp = np.amin(whole_disps)

    # initialize figures
    trajectory_fig, trajectory_axs = plt.subplots(2, len(output_timesteps), figsize=(12, 5), constrained_layout=True)


    # %% Extract data and plot
    for i, trajectory in enumerate(trajectories):

        ## Extract data from runout
        # displacements
        disps = []
        initial_position = trajectory[0, :]
        for current_position in trajectory:
            displacement = initial_position - current_position
            scalr_disp = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
            disps.append(scalr_disp)
        disps = np.array(disps)
        # velocities
        velocities = trajectory[1:, ] - trajectory[:-1, ]  # velocities for x and y
        initial_velocity = velocities[0]  # copy initial velocity
        initial_velocity = np.expand_dims(initial_velocity, 0)  # add third dimension for concat later with velocities
        velocities = np.concatenate((velocities, initial_velocity))  # Concatenate
        scalar_velocities = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2)  # scalar sum of x y velocities
        # runout height (H) and length (L)
        L_t = []
        H_t = []
        for position in trajectory:
            # if mode == "maximum":
            #     L = np.amax(position[:, 0])  # front end of runout
            #     H = np.amax(position[:, 1])  # top of runout
            # elif mode == "95_percentile":
            L = np.percentile(position[:, 0], output_percentile) - np.min(trajectory[0][:, 0])  # front end of runout
            H = np.percentile(position[:, 1], output_percentile) - np.min(trajectory[0][:, 1])  # top of runout
            L_t.append(L)
            H_t.append(H)
        # normalize H, L with initial length of column and time with critical time
        L_initial = np.amax(trajectory[0][:, 0]) - np.min(trajectory[0][:, 0])
        H_initial = np.amax(trajectory[0][:, 1]) - np.min(trajectory[0][:, 1])
        critical_time = np.sqrt(H_initial / g)
        time = np.arange(0, timesteps)  # assume dt=1
        normalized_time = time / critical_time  # critical time assuming dt=1
        normalized_L_t = (L_t - L_initial) / L_initial
        normalized_H_t = H_t / L_initial
        # compute energies
        potentialE = np.sum(mass * g * trajectory[:, :, 1], axis=-1)  # sum(mass * gravity * elevation)
        kineticE = (1 / 2) * np.sum(mass * scalar_velocities ** 2, axis=-1)
        E0 = potentialE[0] + kineticE[0]
        dissipationE = E0 - kineticE - potentialE
        # normalize energies
        normalized_Ek = kineticE / E0
        normalized_Ep = potentialE / E0
        normalized_Ed = dissipationE / E0

        if i == 0:
            print(f"mpm: {normalized_L_t[-1]}")
        if i == 1:
            print(f"gns: {normalized_L_t[-1]}")

        ##  Plot
        # plot for runout
        p1 = runout_ax.plot(normalized_time, normalized_L_t,
                            color=runout_colors[i],
                            linestyle=runout_lines[0],
                            label=runout_legends[0][i])
        p2 = height_ax.plot(normalized_time, normalized_H_t,
                            color=runout_colors[i],
                            linestyle=runout_lines[1],
                            label=runout_legends[1][i])
        # labels
        runout_ax.set_xlabel(r"$t / \tau_{c}$")
        runout_ax.set_ylabel(r"$(L_t - L_0)/L_0$")
        height_ax.set_ylabel(r"$H_t/L_0$")
        runout_ax.set_xlim(xmin=0, xmax=4)
        runout_ax.set_ylim(ymin=0, ymax=3.5)
        height_ax.set_xlim(xmin=0, xmax=4)
        height_ax.set_ylim(ymin=0, ymax=2.5)
        # legend
        # runout_p_set = p1 + p2
        # runout_p_sets.extend(runout_p_set)
        # runout_labs = [l.get_label() for l in runout_p_sets]
        # runout_ax.legend(ncol=2, loc="middle right", prop={'size': 8})
        lines, labels = runout_ax.get_legend_handles_labels()
        lines2, labels2 = height_ax.get_legend_handles_labels()
        height_ax.legend(lines + lines2, labels + labels2,
                          ncol=2, loc="lower right", prop={'size': 8})

        runout_fig.tight_layout()
        # if i == 1:
        #     runout_fig.show()
        #     # runout_fig.savefig(f"Runout-{rollout_filename}_{percentile}percentile.png")

        # plot for energy
        sample_rate = 1
        p3 = energy_ax.plot(normalized_time[::sample_rate], normalized_Ep[::sample_rate],
                            color=energy_colors[0][i],
                            linestyle=energy_lines[0],
                            label=energy_legends[0][i])
        p4 = energy_ax2.plot(normalized_time[::sample_rate], normalized_Ek[::sample_rate],
                                  color=energy_colors[1][i],
                                  linestyle=energy_lines[1],
                                  label=energy_legends[1][i])
        p5 = energy_ax2.plot(normalized_time[::sample_rate], normalized_Ed[::sample_rate],
                                  color=energy_colors[2][i],
                                  linestyle=energy_lines[2],
                                  label=energy_legends[2][i])

        energy_ax.set_xlabel(r"$t / \tau_{c} $")
        energy_ax.set_ylabel(r"$E_p/E_0$")
        energy_ax.set_xlim(xmin=0, xmax=4)
        energy_ax.set_ylim(ymin=0)
        energy_p_set = p3 + p4 + p5
        energy_p_sets.extend(energy_p_set)
        energy_ax2.set_ylabel(r"$E_k/E_0$ and $E_d/E_0$")
        energy_ax2.set_ylim(ymin=0, ymax=0.7)
        # energy_ax2.set_ylim(ymin=0, ymax=0.3)
        # lines, labels = energy_ax.get_legend_handles_labels()
        # lines2, labels2 = energy_ax2.get_legend_handles_labels()
        # energy_ax2.legend(lines + lines2, labels + labels2,
        #                   ncol=3, loc=7, prop={'size': 8})
        # energy_ax.legend(ncol=2, loc="best", prop={'size': 8})
        # energy_ax.legend(ncol=2, loc="best", prop={'size': 8})
        # energy_ax2.legend(ncol=2, loc="best", prop={'size': 8})
        energy_fig.tight_layout()

        # Runout with timesteps
        # norm = colors.TwoSlopeNorm(vmin=min_disp, vcenter=0.05, vmax=max_disp)
        norm = colors.TwoSlopeNorm(vcenter=0.035)

        for j, output_timestep in enumerate(output_timesteps):
            # p = trajectory_axs[i, j].scatter(
            #     trajectory[output_timesteps[j], :, 0], trajectory[output_timesteps[j], :, 1],
            #     vmin=min_disp, vmax=max_disp+0.4,
            #     # vmin=np.min(disps[output_timesteps[j]]), vmax=np.max(disps[output_timesteps[j]]),
            #     s=0.2, c=disps[output_timesteps[j]], cmap='viridis')
            p = trajectory_axs[i, j].scatter(
                trajectory[output_timesteps[j], :, 0], trajectory[output_timesteps[j], :, 1],
                norm=norm,
                # vmin=np.min(disps[output_timesteps[j]]), vmax=np.max(disps[output_timesteps[j]]),
                s=0.2, c=disps[output_timesteps[j]], cmap='viridis') # viridis, bwr
            if i == 0:
                trajectory_axs[i, j].set_title(f"MPM at $t_c$={np.around(output_critical_timesteps[j], 1)}")
            if i==1:
                trajectory_axs[i, j].set_title(f"GNS at $t_c$={np.around(output_critical_timesteps[j], 1)}")
            trajectory_axs[i, j].set_xlim(boundary[0])
            trajectory_axs[i, j].set_ylim(boundary[1])
            # trajectory_axs[i, j].set_xlabel("x")
            # trajectory_axs[i, j].set_ylabel("y")
            trajectory_axs[i, j].set_aspect('equal')
    #
    # trajectory_fig.colorbar(p, ax=trajectory_axs[:, 3], shrink=0.6)
    runout_fig.show()
    runout_fig.savefig(f"{rollout_path}/runout_{output_filename}-{int(output_percentile)}percentile.png")
    # energy_fig.show()
    energy_fig.savefig(f"{rollout_path}/energy_{output_filename}.png")
    # trajectory_fig.show()
    trajectory_fig.savefig(f"{rollout_path}/trajectory_{output_filename}.png")
    # mpm_trajectory_fig.show()
    # gns_trajectory_fig.show()
