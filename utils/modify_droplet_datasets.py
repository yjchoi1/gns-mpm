import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

npz_loc = '/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/droplet2/'
save_loc = "/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/droplet4/"
data = dict(np.load(f"{npz_loc}/train.npz", allow_pickle=True))
boundaries = [[-1.2418140152084631, 46.841814015208456],
              [-5.16455663821927, 9.321478576384568]]
distance = 2
left_bounds = np.arange(boundaries[0][0])
timesteps = data["simulation_trajectory_1"][0].shape[0]
boundary_material_id = 3

def generate_boundary_equal_spacing_coordinates(x_start, x_stop, y_start, y_stop, spacing):
    x_range = x_stop - x_start
    y_range = y_stop - y_start
    x_steps = math.ceil(x_range/spacing)
    y_steps = math.ceil(y_range/spacing)
    x = [x_start + i*(x_range/x_steps) for i in range(x_steps+1)]
    y = [y_start + i*(y_range/y_steps) for i in range(y_steps+1)]
    xy_coordinates = [(x_coord, y_coord) for x_coord in [x_start, x_stop] for y_coord in y] + [(x_coord, y_start) for x_coord in x[1:-1]] + [(x_coord, y_stop) for x_coord in x[1:-1]]
    xy_coordinates = np.array(xy_coordinates)
    return xy_coordinates

# generate boundary particles
boundary_particles = generate_boundary_equal_spacing_coordinates(
    x_start=boundaries[0][0],
    x_stop=boundaries[0][1],
    y_start=boundaries[1][0],
    y_stop=boundaries[1][1],
    spacing=2.5
)

# make it fit to the dimension of the data
boundary_particles = [boundary_particles]*timesteps
boundary_particles = np.array(boundary_particles)

data_modified = {}
# append the boundary particles and material info to the data
for i, (sim, particle) in enumerate(data.items()):
    coords = particle[0]
    coords_with_boundary = np.append(coords, boundary_particles, axis=1)
    material = particle[1]
    material_with_boundary = np.append(material, np.full(boundary_particles.shape[1], boundary_material_id))
    data_modified[f"simulation_trajectory_{i+1}"] = (coords_with_boundary, material_with_boundary)

# save npz for each trajectory
for sim, particle in data_modified.items():
    each_data = {}
    each_data[sim] = (particle[0], particle[1])
    np.savez_compressed(f"{save_loc}{sim}.npz", **each_data)

data_loc = "/work2/08264/baagee/frontera/gns-mpm/gns-data/datasets/droplet4/"
see_data = dict(np.load(f"{data_loc}train.npz", allow_pickle=True))


# visualize and check if the modification is correct
for i, (sim, particle) in enumerate(see_data.items()):
    coords_modified = particle[0]
    material_modified = particle[1]
colors = []
for i in material_modified:
    if i == 0:
        colors.append("b")
    elif i == 1:
        colors.append("r")
    else:
        colors.append("k")

fig, ax = plt.subplots()
t = 231
ax.scatter(coords_modified[t][:, 0], coords_modified[t][:, 1], c=colors)
ax.set_aspect(1)
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