from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import pandas as pd

mesh_shape = "quad"
npz_loc = '/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/lbm-pipe/sampled.npz'
data = [dict(trj_info.item()) for trj_info in np.load(npz_loc, allow_pickle=True).values()]
# data_merged = {}
# for i, d in enumerate(data[0:10]):
#     a_data = {f"sim{i}": d}
#     data_merged.update(a_data)
# np.savez_compressed(
#     f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/lbm-pipe/sampled.npz", **data_merged)

node_type_info = {
  "normal": [0, "gray"],
  "obstacle": [1, "white"],
  "airfoil": [2, "blue"],
  "handle": [3, "yellow"],
  "inflow": [4, "purple"],
  "outflow": [5, "cyan"],
  "wall_boundary": [6, "red"]
}
simulation_region = [[0.0, 1.6], [0.0, 0.4]]
offset = 0.01  # offset from the simulation_region to place obstacle
obstacle_region = [[simulation_region[0][0]+offset, simulation_region[0][1]-offset],
                   [simulation_region[1][0]+offset, simulation_region[1][1]-offset]]

#%%
# See distribution of data
initial_vels = []
obstacles = []
for feature in data:
    # get obstacle center & radius
    node_coord = feature["pos"][0]
    wall_mask = (feature["node_type"][0] == 6)
    x_in_range = (node_coord[:, 0] >= obstacle_region[0][0]) & (node_coord[:, 0] <= obstacle_region[0][1])
    y_in_range = (node_coord[:, 1] >= obstacle_region[1][0]) & (node_coord[:, 1] <= obstacle_region[1][1])
    obstacle_mask = x_in_range & y_in_range & np.squeeze(wall_mask)
    obstacle_node_coords = node_coord[obstacle_mask]
    obstacle_center = [obstacle_node_coords[:, 0].mean(), obstacle_node_coords[:, 1].mean()]
    obstacle_radius = (obstacle_node_coords[:, 0].max() - obstacle_node_coords[:, 0].min())/2
    obstacle_info = np.array([obstacle_center[0], obstacle_center[1], obstacle_radius])
    obstacles.append(obstacle_info)

    # get initial vel
    initial_vel = feature["velocity"][0]
    initial_vels.append(initial_vel)

# get statistics
initial_vels_df = pd.DataFrame(np.concatenate(initial_vels))
obstacle_df = pd.DataFrame(np.stack(obstacles))
initial_vels_df.describe()
obstacle_df.describe()


#%% Plot
data_id = 1
frame = 500

cmap = ListedColormap([node_type_color[1] for node_type_color in node_type_info.values()])
labels = [node_type_name for node_type_name in node_type_info.keys()]

pos = data[data_id]["pos"]  # shape=(600, 1923, 2)
node_type = data[data_id]["node_type"]  # shape=(600, 1923)
cells = data[data_id]["cells"]
vel = data[data_id]["velocity"]  # shape=(600, 1923, 2)
vel_mag = np.linalg.norm(vel, axis=-1)  # shape=(600, 1923, 1)
print(f"nnodes: {len(node_type[0])}")
print(f"ncells: {len(cells[0])}")

if mesh_shape == "triangle":
    triang = tri.Triangulation(pos[0][:, 0], pos[0][:, 1])
    # color
    vmin = np.concatenate((vel[0][:, 0], vel[0][:, 0])).min()
    vmax = np.concatenate((vel[0][:, 0], vel[0][:, 0])).max()

    fig = plt.figure(figsize=(10, 4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 1),
                     axes_pad=0.3,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="1.5%",
                     cbar_pad=0.15)

    handle = grid[0].tripcolor(triang, vel_mag[frame], vmax=vmax, vmin=vmin)
    grid[0].triplot(triang, 'k-', lw=0.1)
    scatter = grid[0].scatter(pos[0][:, 0], pos[0][:, 1], c=node_type[0], s=2.0, cmap=cmap)
    # fig.colorbar(scatter, ax=grid[0], label="Node Type")  # New colorbar for node_type
    fig.colorbar(handle, cax=grid.cbar_axes[0], label="Velocity (m/s)")

    handles = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))]
    grid[0].legend(handles=handles, loc='best')

    plt.show()

if mesh_shape == "quad":
    # color
    vmin = vel_mag.min()
    vmax = vel_mag.max()

    fig = plt.figure(figsize=(10, 4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 1),
                     axes_pad=0.3,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="1.5%",
                     cbar_pad=0.15)

    # Reshape data to fit my lbm grid structure. Current grid: (160 * 40)
    vel_mag_grid = vel_mag[frame].reshape(41, 161)
    x_grid = pos[0][:, 0].reshape(41, 161)
    y_grid = pos[0][:, 1].reshape(41, 161)

    # Setting contour levels between min and max values of magnitudes
    levels = np.linspace(vel_mag.min(), vel_mag.max(), 100)

    # make the contour plot
    velocity_contour = grid[0].contourf(x_grid, y_grid, vel_mag_grid, cmap='viridis', levels=levels)
    node_type = grid[0].scatter(pos[0][:, 0], pos[0][:, 1], c=node_type[0], s=2.0, cmap=cmap)
    fig.colorbar(velocity_contour, cax=grid.cbar_axes[0])
    handles = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))]
    grid[0].legend(handles=handles, loc='best')
    plt.show()
