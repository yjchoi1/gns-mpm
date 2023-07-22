import torch
import sys
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/meshnet/')
from meshnet.utils import NodeType
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def get_node_type(node_coord,
                  node_type_sim_domain,
                  current_obstacle_location,
                  radius_tol=0.0001):
    """
    Create `current_node_type` tensor considering the current center and radius of circular obstacle.

    Args:
        node_coord: x,y coordinate of nodes with shape=(nnodes, 2)
        node_type_sim_domain: node types for simulation domain without obstacle with shape=(nnodes, 1)
        obstacle_location: mask representing the place where obstacle should be placed in node_type

    Returns:
        Node_type with reflecting current obstacle
    """

    center = current_obstacle_location[0:2]
    radius = current_obstacle_location[2]
    nnode = node_coord.shape[0]

    # First,get `current_obstacle_mask`: true if `node_coord` is in the circular obstacle
    distances = torch.sqrt(torch.sum((node_coord - center) ** 2, dim=1))
    current_obstacle_mask = (distances <= radius + radius_tol)
    # Next, make `current_node_type` tensor:
    # fill with `WALL_BOUNDARY` if `current_obstacle_mask`, else, maintain `node_type_sim_domain`.
    current_node_type = torch.where(
        current_obstacle_mask,
        torch.full((nnode, 1), NodeType.WALL_BOUNDARY).squeeze(),
        node_type_sim_domain.squeeze()
    ).to(torch.float32)

    return current_node_type


def plot_flow(results, timestep, save_path):

    # Preprocess results for plotting
    result_processed = {
        "ground_truth": {
            "vel_mag": np.linalg.norm(results["ground_truth"]["vel"], axis=-1),
            "wall_mask": results["ground_truth"]["node_type"] == NodeType.WALL_BOUNDARY
        },
        "prediction": {
            "vel_mag": np.linalg.norm(results["prediction"]["vel"], axis=-1),
            "wall_mask": results["prediction"]["node_type"] == NodeType.WALL_BOUNDARY
        }
    }

    n_timesteps = len(result_processed["ground_truth"]["vel_mag"])
    triang = tri.Triangulation(results["pos"][0][:, 0], results["pos"][0][:, 1])

    # color
    vmin = result_processed["ground_truth"]["vel_mag"].min()
    vmax = result_processed["ground_truth"]["vel_mag"].max()

    # init figure
    fig = plt.figure(figsize=(10, 4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 1),
                     axes_pad=0.3,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="1.5%",
                     cbar_pad=0.15)

    # plot
    for j, (sim, data) in enumerate(result_processed.items()):
        print(sim)
        grid[j].triplot(triang, '-', color='k', lw=0.3)
        handle = grid[j].tripcolor(triang, data["vel_mag"][timestep-1], vmax=vmax, vmin=vmin)
        grid[j].scatter(
            results["pos"][0][:, 0][data["wall_mask"].squeeze()],
            results["pos"][0][:, 1][data["wall_mask"].squeeze()], s=0.5, c="red", zorder=10)
        fig.colorbar(handle, cax=grid.cbar_axes[0])
        grid[j].set_title(sim)

    # save fig
    plt.savefig(f"{save_path}")
