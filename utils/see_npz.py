import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


data = np.load('/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/droplet2/merged_train2d.npz', allow_pickle=True)
output = "."

for i, p in data.items():
    print(i, p)
    print(np.shape(p[0]))
    positions = p[0]
    particles = p[1]

times = range(len(positions[0]))
nparticles = len(positions[1])
xpositions = []


# ax = fig.add_subplot(projection='3d')
for i, p in enumerate(positions[::100]):
    xposition = p[:, 0]
    yposition = p[:, 1]
    zposition = p[:, 2]
    # if i ==0 or i == 100:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xposition, yposition, zposition)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"timestep={i}")
    plt.show()


# # make animation
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# def animate(i):
#     fig.clear()
#     xboundary = [0, 40]
#     yboundary = [-0.04, 0.04]
#     zboundary = [0, 4]
#     # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
#     ax = fig.add_subplot(projection='3d', autoscale_on=False)
#     ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
#     ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
#     ax.set_zlim([float(zboundary[0]), float(zboundary[1])])
#     ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2], s=5)
#     ax.view_init(elev=20., azim=i*0.2)
#     ax.grid(True, which='both')
#
# ani = animation.FuncAnimation(
#     fig, animate, frames=np.arange(0, len(positions), 3), interval=100)
#
# ani.save(f'{output}/trajectory.gif', dpi=100, fps=30, writer='imagemagick')
# print(f"Animation saved to: {output}")