import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import os
import sys
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


test_rollout_tags = ["test4-3"]
data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand-small-r300-400step_serial/"
training_steps = 15270000
train_trajectory_ID = [0]

def get_simulations(data_path, rollout_tags, trajectory_ID, training_steps=None):
    # import rollout.pkl
    rollout_filenames = []
    for rollout_tag in rollout_tags:
        for i in trajectory_ID:
            if training_steps is not None:
                rollout_filename = f"rollout_{rollout_tag}_{i}_step{training_steps}"
            else:
                rollout_filename = f"rollout_{rollout_tag}_{i}"
            rollout_filenames.append(rollout_filename)
    # print(rollout_filenames)
    trajectories = {"mpm": {}, "gns": {}}
    metadatas = {}
    for rollout_filename in rollout_filenames:
        with open(os.path.join(data_path, rollout_filename + ".pkl"), "rb") as file:
            # load rollout
            rollout = pickle.load(file)
            metadatas[rollout_filename] = rollout["metadata"]
            # get trajectories
            gns_trajectory = np.concatenate([rollout["initial_positions"], rollout["predicted_rollout"]], axis=0)
            mpm_trajectory = np.concatenate([rollout["initial_positions"], rollout["ground_truth_rollout"]], axis=0)
            trajectories["mpm"][rollout_filename] = mpm_trajectory
            trajectories["gns"][rollout_filename] = gns_trajectory

    velocities = {
        "mpm": {"x": {}, "y": {}, "magnitude": {}},
        "gns": {"x": {}, "y": {}, "magnitude": {}}
    }
    flattened_velocities = {
        "mpm": {"x": {}, "y": {}, "magnitude": {}},
        "gns": {"x": {}, "y": {}, "magnitude": {}}
    }
    for simulation_method, trajectories_of_simulation in trajectories.items():
        for id, trajectory in trajectories_of_simulation.items():
            velocity = trajectory[1:, ] - trajectory[:-1, ]  # velocity for x and y
            initial_velocity = velocity[0]  # copy initial velocity
            initial_velocity = np.expand_dims(initial_velocity,
                                              0)  # add third dimension for concat later with velocities
            velocity = np.concatenate((velocity, initial_velocity))  # Concatenate
            velocity_magnitude = np.sqrt(
                velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)  # scalar sum of x y velocities
            velocities[f"{simulation_method}"]["x"][id] = velocity[:, :, 0]  # (320, 2304)
            # print(np.shape(velocity[:, :, 0]))
            velocities[f"{simulation_method}"]["y"][id] = velocity[:, :, 1]  # (320, 2304)
            # print(np.shape(velocity[:, :, 1]))
            velocities[f"{simulation_method}"]["magnitude"][id] = velocity_magnitude  # (320, 2304)
            # print(np.shape(velocity_magnitude))
            flattened_velocities[f"{simulation_method}"]["x"][id] = np.ndarray.flatten(
                velocity[:, :, 0])
            # print(np.shape(np.ndarray.flatten(velocity[:, :, 0])))
            flattened_velocities[f"{simulation_method}"]["y"][id] = np.ndarray.flatten(
                velocity[:, :, 1])
            # print(np.shape(np.ndarray.flatten(velocity[:, :, 1])))
            flattened_velocities[f"{simulation_method}"]["magnitude"][id] = np.ndarray.flatten(
                velocity_magnitude)

    return trajectories, velocities, flattened_velocities, metadatas

def whole_velocity_distribution(flattened_velocities, simulation):
    whole_velocity_distribution = {}
    for base, flattened_velocity in flattened_velocities[simulation].items():
        velocity_set = []
        for velocity_for_trajectory in flattened_velocity.values():
            velocity_set.append(velocity_for_trajectory)
        whole_velocity_distribution[base] = np.concatenate(velocity_set)
    return whole_velocity_distribution

def bin_and_prob(data, bin_start, bin_end, nbins):
    '''
    Get probability and bins for plotting bar plot
    :param data:
    :param bin_start:
    :param bin_end:
    :param nbins:
    :return:
    '''
    bin_edges = np.linspace(start=bin_start, stop=bin_end, num=nbins + 1, endpoint=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_start_edges = bin_edges[:-1]
    freqs, _ = np.histogram(data, bins=bin_edges)
    to_probability = 1 / len(data)
    probabilities = freqs * to_probability
    print(f"Total sum of PMF: {sum(probabilities)}")
    return probabilities, bin_start_edges, bin_width

train_trajectories, train_velocities, flattened_train_velocities, train_metadatas = get_simulations(
    data_path=data_path,
    rollout_tags=["train"],
    trajectory_ID=train_trajectory_ID,
    training_steps=training_steps
)

test_trajectories, test_velocities, flattened_test_velocities, test_metadatas = get_simulations(
    data_path=data_path,
    rollout_tags=test_rollout_tags,
    trajectory_ID=[0],
    training_steps=training_steps
)

whole_train_vel_distb = whole_velocity_distribution(
    flattened_velocities=flattened_train_velocities, simulation="mpm"
)

# Get max min vels for color plots in entire simulations (both mpm and gns) to
test_vels = []
for simulation in ["mpm", "gns"]:
    whole_test_vel_distb = whole_velocity_distribution(
        flattened_velocities=flattened_test_velocities, simulation=simulation
    )
    test_vels.append(whole_test_vel_distb['magnitude'])
concat_test_vels = np.concatenate(test_vels)
max_test_vel = np.max(concat_test_vels)
min_test_vel = np.min(concat_test_vels)

###

###

#%% Plots

# Inputs for plot
# number of bins
nbins = 100
# range for bins
first_edge, last_edge = whole_train_vel_distb['magnitude'].min(), whole_train_vel_distb['magnitude'].max()
# timesteps to plot
critical_timesteps = [0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
# get test_rollout file names to plot
test_rollout_filenames = []
trajectory_ID = [0]
# figsize
figsize_trajectory = (8, 2)
# title
title_loc = 0.85
title_fontsize = 10
# color bar
colorbar = {"aspect": 30, "labelsize": 8}

for test_rollout_tag in test_rollout_tags:
    for i in trajectory_ID:
        test_rollout_filename = f"rollout_{test_rollout_tag}_{i}_step{training_steps}"
        test_rollout_filenames.append(test_rollout_filename)

# disp for color
for test_rollout in test_rollout_filenames:
    # prepare test distribution and overlap to the train distribution plot
    timesteps = np.shape(test_velocities['gns']['magnitude'][test_rollout])[0]
    # timesteps_to_plot = np.linspace(0, timesteps, num=nsample_timesteps, endpoint=True, dtype=int)
    ### Replace timesteps_to_plot from "explicit time" to "normazlied time"
    dt = test_metadatas[test_rollout]["dt"]
    g = 9.81 * dt ** 2  # because, in MPM, we used dt=0.0025, and, in GNS, dt=1.0
    height = np.max(test_trajectories["mpm"][test_rollout][0][:, 1]) - np.min(test_trajectories["mpm"][test_rollout][0][:, 1])
    critical_time = np.sqrt(height / g)
    output_critical_timesteps = critical_timesteps + [timesteps / critical_time]
    output_timesteps = np.around(np.array(output_critical_timesteps) * critical_time, 0)
    output_timesteps = list(output_timesteps.astype(int))
    output_timesteps[-1] = timesteps  # to comply with the last index of the list
    timesteps_to_plot = output_timesteps
    timesteps_to_plot[-1] = timesteps_to_plot[-1] - 1  # exclude the last index for consider dimension

    # Disp for color
    for i, timestep in enumerate(timesteps_to_plot):
        disp_x = test_trajectories["mpm"][test_rollout][0][:, 0]-test_trajectories["mpm"][test_rollout][timestep][:, 0]
        disp_y = test_trajectories["mpm"][test_rollout][0][:, 1]-test_trajectories["mpm"][test_rollout][timestep][:, 1]



# prepare train distribution and plot
# train distribution
p_train_m, bin_start_edges, bin_width = bin_and_prob(
    data=whole_train_vel_distb['magnitude'], bin_start=first_edge, bin_end=last_edge, nbins=nbins)
# test distribution
p_test_m, bin_test_start_edges, bin_test_width = bin_and_prob(
    data=whole_test_vel_distb['magnitude'], bin_start=first_edge, bin_end=last_edge, nbins=nbins)
percentile = int(stats.percentileofscore(bin_start_edges, max_test_vel))

# plot
# 1) displacement and velocity trajectory,
# 2) velocity distribution compare to train velocity distribution,
# for each defined timestep
for test_rollout in test_rollout_filenames:
    # prepare test distribution and overlap to the train distribution plot
    timesteps = np.shape(test_velocities['gns']['magnitude'][test_rollout])[0]
    # timesteps_to_plot = np.linspace(0, timesteps, num=nsample_timesteps, endpoint=True, dtype=int)

    ### Replace timesteps_to_plot from "explicit time" to "normazlied time"
    dt = test_metadatas[test_rollout]["dt"]
    g = 9.81 * dt ** 2  # because, in MPM, we used dt=0.0025, and, in GNS, dt=1.0
    height = np.max(test_trajectories["mpm"][test_rollout][0][:, 1]) - np.min(test_trajectories["mpm"][test_rollout][0][:, 1])
    critical_time = np.sqrt(height / g)
    output_critical_timesteps = critical_timesteps + [timesteps / critical_time]
    output_timesteps = np.around(np.array(output_critical_timesteps) * critical_time, 0)
    output_timesteps = list(output_timesteps.astype(int))
    output_timesteps[-1] = timesteps  # to comply with the last index of the list
    timesteps_to_plot = output_timesteps
    timesteps_to_plot[-1] = timesteps_to_plot[-1] - 1  # exclude the last index for consider dimension

    # Disp magnitude for color
    disp_magnitudes = {"mpm": [], "gns": []}
    for simulation in disp_magnitudes.keys():
        for i, timestep in enumerate(timesteps_to_plot):
            disp_x = test_trajectories[simulation][test_rollout][0][:, 0]-test_trajectories[simulation][test_rollout][timestep][:, 0]
            disp_y = test_trajectories[simulation][test_rollout][0][:, 1]-test_trajectories[simulation][test_rollout][timestep][:, 1]
            disp_magnitude = np.sqrt(disp_x ** 2 + disp_y ** 2)
            disp_magnitudes[simulation].append(disp_magnitude)
    whole_disp_mags = {"mpm": [], "gns": []}
    for simulation in whole_disp_mags.keys():
        whole_disp_mags[simulation] = np.concatenate(disp_magnitudes[simulation])
    merged_disp_mags =  np.concatenate((whole_disp_mags["mpm"], whole_disp_mags["gns"]))

    # for timestep in timesteps_to_plot:
    for i, timestep in enumerate(timesteps_to_plot):
        fig_disp, axd = plt.subplot_mosaic([['mpm_disp', 'gns_disp']],
                                      figsize=figsize_trajectory, sharex=True, layout="constrained")
        fig_vel, axv = plt.subplot_mosaic([['mpm_vel', 'gns_vel']],
                                      figsize=(6, 1.5), sharex=True, layout="constrained")
        fig_distb, ax_distb = plt.subplots(figsize=(3, 2.5), layout="constrained")

        # Trajectory plots
        bounds = test_metadatas[test_rollout]["bounds"]

        # trajectory plot - displacement contour
        # norm = colors.TwoSlopeNorm(vcenter=0.035)  # TODO: `vcenter` value need to be determined
        # mpm displacement
        d_mpm = axd['mpm_disp'].scatter(
            test_trajectories["mpm"][test_rollout][timestep][:, 0], test_trajectories["mpm"][test_rollout][timestep][:, 1],
            s=2, vmin=np.min(merged_disp_mags), vmax=np.max(merged_disp_mags),
            c=disp_magnitudes["mpm"][i])
            # s=2, c=disp_magnitude, norm=norm, cmap='viridis')
        divider = make_axes_locatable(axd['mpm_disp'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig_disp.colorbar(d_mpm, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=colorbar["labelsize"])
        axd['mpm_disp'].set_xlim(bounds[0])
        axd['mpm_disp'].set_ylim(bounds[1][0], bounds[1][1])
        axd['mpm_disp'].set_title("MPM", y=title_loc, fontsize=title_fontsize)
        axd['mpm_disp'].set_aspect('equal')
        fig_disp.tight_layout()

        # gns displacement
        d_gns = axd['gns_disp'].scatter(
            test_trajectories["gns"][test_rollout][timestep][:, 0], test_trajectories["gns"][test_rollout][timestep][:, 1],
            s=2, vmin=np.min(merged_disp_mags), vmax=np.max(merged_disp_mags),
            c=disp_magnitudes["gns"][i])
            # s=2, c=disp_magnitude, norm=norm, cmap='viridis')
        divider = make_axes_locatable(axd['gns_disp'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig_disp.colorbar(d_gns, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=colorbar["labelsize"])
        axd['gns_disp'].set_xlim(bounds[0])
        axd['gns_disp'].set_ylim(bounds[1][0], bounds[1][1])
        axd['gns_disp'].set_title("GNS", y=title_loc, fontsize=title_fontsize)
        axd['gns_disp'].set_aspect('equal')
        plt.tight_layout()

        # trajectory plot - velocity contour
        norm = colors.TwoSlopeNorm(vcenter=0.0005)  # TODO: `vcenter` value need to be determined
        # mpm velocity
        v_mpm = axv['mpm_vel'].scatter(
            test_trajectories["mpm"][test_rollout][timestep][:, 0], test_trajectories["mpm"][test_rollout][timestep][:, 1],
            s=2, vmin=min_test_vel, vmax=max_test_vel, c=test_velocities["mpm"]["magnitude"][test_rollout][timestep])
            # s=0.3, c=test_velocities["mpm"]["magnitude"][test_rollout][timestep], norm=norm, cmap='viridis')
        divider = make_axes_locatable(axv['mpm_vel'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig_vel.colorbar(v_mpm, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=colorbar["labelsize"])
        axv['mpm_vel'].set_xlim(bounds[0])
        axv['mpm_vel'].set_ylim(bounds[1][0], bounds[1][1])
        axv['mpm_vel'].set_title("MPM", y=title_loc, fontsize=title_fontsize)
        axv['mpm_vel'].set_aspect('equal')
        plt.tight_layout()
        # gns velocity
        v_gns = axv['gns_vel'].scatter(
            test_trajectories["gns"][test_rollout][timestep][:, 0], test_trajectories["gns"][test_rollout][timestep][:, 1],
            s=2, vmin=min_test_vel, vmax=max_test_vel, c=test_velocities["gns"]["magnitude"][test_rollout][timestep])
            # s=0.3, c=test_velocities["gns"]["magnitude"][test_rollout][timestep], norm=norm, cmap='viridis')
        divider = make_axes_locatable(axv['gns_vel'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig_vel.colorbar(v_gns, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=colorbar["labelsize"])
        axv['gns_vel'].set_xlim(bounds[0])
        axv['gns_vel'].set_ylim(bounds[1][0], bounds[1][1])
        axv['gns_vel'].set_title("GNS", y=title_loc, fontsize=title_fontsize)
        axv['gns_vel'].set_aspect('equal')
        plt.tight_layout()
        # get color values and divide it into the equal length corresponding to the bins
        v_colors = v_gns.cmap.colors
        v_colors_bins = [v_colors[int(i)] for i in np.linspace(0, len(v_colors), percentile, endpoint=False)]



        # Test distribution - use the same bin start and end range with the train distribution
        p_test_m, _, _ = bin_and_prob(
            data=test_velocities['gns']['magnitude'][test_rollout][timestep],
            bin_start=first_edge, bin_end=last_edge, nbins=nbins)
        ax_distb.bar(x=bin_start_edges, height=p_train_m,
                           width=bin_width, align='edge', alpha=0.5, color="black", label="Entire train")
        ax_distb.bar(x=bin_start_edges, height=p_test_m,
                           width=bin_width, align='edge', alpha=0.5, color=v_colors_bins, label="Test")
        ax_distb.set_xlim([0, 0.0055])
        ax_distb.set_ylim([10**(-2), 1.0])
        ax_distb.set_yscale('log')
        ax_distb.set_xlabel("Velocity Magnitude (m/s)")
        ax_distb.set_ylabel("Probability")
        ax_distb.set_title("Distribution of velocity")
        ax_distb.legend()

        # Save plots
        # fig.suptitle(f"{test_rollout} ($t_c$={output_critical_timesteps[i]:.1f})")
        fig_disp.savefig(f"{data_path}/disp_{test_rollout}_(tc={output_critical_timesteps[i]:.1f}).png")
        fig_disp.show()
        
        fig_vel.savefig(f"{data_path}/vel_{test_rollout}_(tc={output_critical_timesteps[i]:.1f}).png")
        fig_vel.show()

        fig_distb.savefig(f"{data_path}/distb_{test_rollout}_(tc={output_critical_timesteps[i]:.1f}).png")
        fig_distb.show()