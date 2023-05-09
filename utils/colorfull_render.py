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
from error_analysis import get_positions, compute_kinemacis
from mpl_toolkits.axes_grid1 import ImageGrid


rollout_tags = \
    [
    # [("test2-1", "a=0.5", "solid", "0.9"),  # (simulation id, aspect ratio, linetype, linecolor)
    #  ("test4", "a=0.8", (0, (5, 1)), "0.7"),
    #  ("test0-2", "a=1.0", (0, (5, 5)), "0.5"),
    #  ("test5-1", "a=2.0", (0, (1, 1)), "0.3")]
    #  ("test5-5", "a=3.0", "dashdot", "0.15"),
    #  ("test6-1", "a=4.0", (0, (3, 5, 1, 5)), "0.0"),
     ("eval12", "a=0.5", "dotted", "0.7")]

gen_animation = False
gen_plots = True
if gen_plots == True:
    normalized_time_to_plot = [0, 1.0, 2.5]  # normalized times at which plots are generated
data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand2d_frictions-r015/"
save_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand2d_frictions-r015/"
training_steps = 5000000
trajectory_ID = [0]
mpm_dt = 0.0025
gravity = 9.81
xboundary = [0.0, 1.0]
yboundary = [0.0, 0.5]
timestep_stride = 3


rollout_filenames = []
for tag in rollout_tags:
    for i in trajectory_ID:
        if training_steps is not None:
            rollout_filename = f"rollout_{tag[0]}_{i}_step{training_steps}"
        else:
            rollout_filename = f"rollout_{tag[0]}_{i}"
        rollout_filenames.append(rollout_filename)

# get data
trajectories, metadata = get_positions(data_path, rollout_filenames)
kinematics_info = compute_kinemacis(trajectories)


# make flow geometry evolution plots
if gen_plots is True:
    for ((rollout_name, kin_data), (rollout_name, trj_data)) in zip(kinematics_info.items(), trajectories.items()):
        # get timesteps that corresponds to normalized times
        num_steps = len(trj_data["mpm"]["positions"])
        final_time = num_steps * mpm_dt
        L0 = trj_data["mpm"]["positions"][0][:, 0].max() - trj_data["mpm"]["positions"][0][:, 0].min()
        H0 = trj_data["mpm"]["positions"][0][:, 1].max() - trj_data["mpm"]["positions"][0][:, 1].min()
        critical_time = np.sqrt(H0 / 9.81)
        times_to_plot = np.array(normalized_time_to_plot) * critical_time
        timesteps_to_plot = np.round(times_to_plot / mpm_dt).astype(int)
        timesteps_to_plot = np.append(timesteps_to_plot, num_steps-1)

        for i in timesteps_to_plot:

            fig = plt.figure(figsize=(5, 2.0))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 2),
                             axes_pad=0.3,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="5%",
                             cbar_pad=0.05
                             )

            for j, (datacase, kinematics) in enumerate(kin_data.items()):
                # sample color values
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(kinematics["displacement"]).max()
                vmin = np.ndarray.flatten(kinematics["displacement"]).min()
                sampled_value = kinematics["displacement"][i]

                # select ax to plot at set boundary
                grid[j].set_aspect(1.)
                grid[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                grid[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                trj = grid[j].scatter(trajectories[rollout_name][datacase]["positions"][i][:, 0],
                                      trajectories[rollout_name][datacase]["positions"][i][:, 1],
                                      c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=2)

                # axes[j].grid(True, which='both')
                grid[j].set_title(datacase)
            fig.colorbar(trj, cax=grid.cbar_axes[0])


            plt.savefig(f"{save_path}/{rollout_name}_time{i}.png")
            plt.tight_layout()
            plt.show()



# make animation
if gen_animation is True:
    for i, (rollout_name, data) in enumerate(kinematics_info.items()):

        num_steps = kinematics_info[rollout_name]["gns"]["displacement"].shape[0]

        # init fig
        fig = plt.figure(figsize=(10, 3.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='rectilinear', autoscale_on=False)
        ax2 = fig.add_subplot(1, 2, 2, projection='rectilinear', autoscale_on=False)
        axes = [ax1, ax2]

        # plot each frame
        def animate(i):
            print(f"Render step {i}/{num_steps}")

            fig.clear()
            for j, (datacase, kinematics) in enumerate(data.items()):
                # sample color values
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(kinematics["displacement"]).max()
                vmin = np.ndarray.flatten(kinematics["displacement"]).min()
                sampled_value = kinematics["displacement"][i]


                # select ax to plot at set boundary
                axes[j] = fig.add_subplot(1, 2, j + 1, autoscale_on=False)
                axes[j].set_aspect(1.)
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                trj = axes[j].scatter(trajectories[rollout_name][datacase]["positions"][i][:, 0],
                           trajectories[rollout_name][datacase]["positions"][i][:, 1],
                           c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=2)
                fig.colorbar(trj, ax=axes[j], shrink=0.5)
                # axes[j].grid(True, which='both')
                axes[j].set_title(datacase)
                plt.tight_layout()


        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, num_steps, timestep_stride), interval=10)

        ani.save(f'{save_path}/{rollout_name}.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {save_path}")