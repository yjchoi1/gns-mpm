import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import animation
from scipy.interpolate import griddata

simulation_name = "trj1"
path = f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/inverse/prac/{simulation_name}/"

with open(f'{path}/results.pkl', 'rb') as handle:
    result = pickle.load(handle)
ground_truth_vel = result["ground_truth"]["vel"]
ground_truth_node_type = result["ground_truth"]["node_type"]
predicted_vel = result["prediction"]["vel"]
prediction_node_type = result["prediction"]["node_type"]

# compute velocity magnitude
ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
processed_result = {
    "ground_truth": (ground_truth_node_type, ground_truth_vel_magnitude),
    "prediction": (ground_truth_node_type, predicted_vel_magnitude)
}

# variables for render
n_timesteps = len(ground_truth_vel_magnitude)
triang = tri.Triangulation(result["pos"][0][:, 0], result["pos"][0][:, 1])

# color
vmin = ground_truth_vel_magnitude.min()
vmax = ground_truth_vel_magnitude.max()

# init figure
fig = plt.figure(figsize=(9.75, 3))


grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 1),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="1.5%",
                 cbar_pad=0.15)

plot_timestep = 20
wall_node = 6
obstacle_region = [[0.0, 1.6], [0.0, 0.4]]
node_coord = result["pos"][0]
each_result_vel = []
for j, (sim, each_result) in enumerate(processed_result.items()):
    # (Find true obstacle x, y, r)
    node_type = each_result[0]
    wall_mask = node_type[0] == wall_node
    wall_mask = np.squeeze(wall_mask)
    x_in_range = (node_coord[:, 0] >= obstacle_region[0][0]) & (node_coord[:, 0] <= obstacle_region[0][1])
    y_in_range = (node_coord[:, 1] >= obstacle_region[1][0]) & (node_coord[:, 1] <= obstacle_region[1][1])
    obstacle_mask = x_in_range & y_in_range & wall_mask
    obstacle_node_coords = node_coord[obstacle_mask]

    vel = each_result[1]
    grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
    handle = grid[j].tripcolor(triang, vel[plot_timestep], vmax=vmax, vmin=vmin)
    grid[j].scatter(obstacle_node_coords[:, 0], obstacle_node_coords[:, 1], c="red", s=1.0)
    fig.colorbar(handle, cax=grid.cbar_axes[0])
    grid[j].set_title(sim)
    each_result_vel.append(vel[plot_timestep])
plt.show()
loss = np.sqrt(np.mean((each_result_vel[0] - each_result_vel[1])**2))
a=1
