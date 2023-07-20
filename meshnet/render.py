import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import animation
from scipy.interpolate import griddata


flags.DEFINE_string("rollout_dir", "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/rollouts/lbm-pipe/", help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", "rollout_0", help="Name of rollout `.pkl` file")
flags.DEFINE_string("mesh_type", "quad", help="Mesh type, either `triangle` or `quad`")
flags.DEFINE_integer("step_stride", 5, help="Stride of steps to skip.")
FLAGS = flags.FLAGS

def render_gif_animation():

    rollout_path = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.pkl"
    animation_filename = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.gif"

    # read rollout data
    with open(rollout_path, 'rb') as f:
        result = pickle.load(f)
    ground_truth_vel = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
    predicted_vel = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))

    # compute velocity magnitude
    ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
    predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
    velocity_result = {
        "ground_truth": ground_truth_vel_magnitude,
        "prediction": predicted_vel_magnitude
    }

    # variables for render
    n_timesteps = len(ground_truth_vel_magnitude)
    if FLAGS.mesh_type == "triangle":
        triang = tri.Triangulation(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])

    # color
    vmin = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).min()
    vmax = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).max()

    # Init figures
    fig = plt.figure(figsize=(9.75, 3))

    def animate(i):
        print(f"Render step {i}/{n_timesteps}")

        fig.clear()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 1),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="1.5%",
                         cbar_pad=0.15)

        if FLAGS.mesh_type == "triangle":
            for j, (sim, vel) in enumerate(velocity_result.items()):
                grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
                handle = grid[j].tripcolor(triang, vel[i], vmax=vmax, vmin=vmin)
                fig.colorbar(handle, cax=grid.cbar_axes[0])
                grid[j].set_title(sim)
        if FLAGS.mesh_type == "quad":
            for j, (sim, vel) in enumerate(velocity_result.items()):

                # X, Y = np.meshgrid(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])
                # # Determine the number of unique x and y values
                # nx = len(result["node_coords"][0][:, 0])
                # ny = len(result["node_coords"][0][:, 1])
                # # Reshape the velocity array
                # Z = vel[i].reshape((nx, ny))

                # create a grid of points on which to interpolate
                grid_x, grid_y = np.mgrid[
                                 min(result["node_coords"][0][:, 0]):max(result["node_coords"][0][:, 0]):100j,
                                 min(result["node_coords"][0][:, 1]):max(result["node_coords"][0][:, 1]):100j]

                # interpolate the velocities onto this grid
                grid_velocity = griddata(result["node_coords"][0], vel[i], (grid_x, grid_y), method='cubic')

                # make the contour plot
                handle = grid[j].contourf(grid_x, grid_y, grid_velocity, 50, vmax=vmax, vmin=vmin, cmap='viridis')
                fig.colorbar(handle, cax=grid.cbar_axes[0])
                grid[j].set_title(sim)

                # handle = grid[j].contourf(X, Y, Z, 50, vmax=vmax, vmin=vmin, cmap='viridis')
                # fig.colorbar(handle, cax=grid.cbar_axes[0])
                # grid[j].set_title(sim)
                # plt.xlabel('X')
                # plt.ylabel('Y')





    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, n_timesteps, FLAGS.step_stride), interval=20)

    ani.save(f'{animation_filename}', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {animation_filename}")


def main(_):
    render_gif_animation()


if __name__ == '__main__':
    app.run(main)
