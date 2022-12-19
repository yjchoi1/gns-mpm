import numpy as np
from matplotlib import pyplot as plt

data = np.load('/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/NateDrop/backup_original_files/test1.npz', allow_pickle=True)

for i, p in data.items():
    print(i, p)
    print(np.shape(p[0]))
    positions = p[0]
    particles = p[1]

times = range(len(positions[0]))
nparticles = len(positions[1])
xpositions = []


# ax = fig.add_subplot(projection='3d')
for i, p in enumerate(positions):
    xposition = p[:, 0]
    yposition = p[:, 1]
    zposition = p[:, 2]
    if i ==0 or i == 100:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xposition, yposition, zposition)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"timestep={i}")
        plt.show()


