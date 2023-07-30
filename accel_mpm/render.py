import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def render_gns_to_mpm(
        data, boundaries, output_name, particle_size=1.0, color="blue", timestep_stride=5):

    fig = plt.figure(figsize=(10, 3.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='rectilinear')
    ax2 = fig.add_subplot(1, 3, 2, projection='rectilinear')
    ax3 = fig.add_subplot(1, 3, 3, projection='rectilinear')
    axes = [ax1, ax2, ax3]
    titles = ["MPM-only", "GNS-only", "GNS+MPM"]
    num_steps = len(data["mpm_rollout"])

    def animate(i):
        if i % 20 == 0:
            print(f"Render step {i}/{num_steps}...")

        fig.clear()
        for j, (datacase, positions) in enumerate(data.items()):
            fig.suptitle(f'{i}/{num_steps}', fontsize=10)
            # select ax to plot at set boundary
            axes[j] = fig.add_subplot(1, 3, j + 1, autoscale_on=False)
            axes[j].scatter(positions[i][:, 0],
                            positions[i][:, 1], s=particle_size, color=color)
            axes[j].set_aspect(1.)
            axes[j].set_xlim([float(boundaries[0][0]), float(boundaries[0][1])])
            axes[j].set_ylim([float(boundaries[1][0]), float(boundaries[1][1])])
            axes[j].grid(True, which='both')
            axes[j].set_xlabel("x (m)")
            axes[j].set_ylabel("y (m)")
            axes[j].set_title(titles[j])
            plt.tight_layout()

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, num_steps, timestep_stride), interval=10)

    ani.save(f'{output_name}', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {output_name}")
