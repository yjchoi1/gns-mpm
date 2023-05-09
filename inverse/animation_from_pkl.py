import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation


def animation_from_pkl(path, output, xbound, ybound):
    """

    Args:
        path: path of `.pkl` file
        output: path to save animation
        xbound: xboundary described by list. e.g., [0, 1]
        ybound: yboundary described by list. e.g., [0, 1]

    Returns:
        Animation
    """
    print("Rendering rollout...")

    # import pkl and get positions
    rollout_path = path
    rollout = pickle.load(open(rollout_path, "rb"))
    positions = rollout["predicted_rollout"]
    ndim = positions.shape[-1]

    # make animation
    if ndim == 2:
        # make animation
        fig, ax = plt.subplots()

        def animate(i):
            fig.clear()
            xboundary = xbound
            yboundary = ybound
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
            ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
            ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], s=1)
            ax.grid(True, which='both')

        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, len(positions), 3), interval=10)

        ani.save(f'{output}/trajectory.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {output}")
    else:
        raise NotImplementedError("Does not support 3d render!")


def plot_final_position(path, output, xbound, ybound):
    rollout_path = path
    final_position = pickle.load(open(rollout_path, "rb"))

    fig, ax = plt.subplots()
    ax.scatter(final_position[:, 0], final_position[:, 1], s=1)
    ax.set_xlim([float(xbound[0]), float(xbound[1])])
    ax.set_ylim([float(ybound[0]), float(ybound[1])])
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    plt.savefig(f"{output}/final_position.png")
    print(f"Plot saved to: {output}")