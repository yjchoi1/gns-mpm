import numpy as np
import math
import json
import argparse
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
    #%% Inputs for mesh, node, node_sets
    parser = argparse.ArgumentParser(description='Make mesh and particle inputs to run mpm')
    parser.add_argument('--ndim', default=2, help="Dimension of simulation domain.")
    parser.add_argument('--x_bounds', nargs='+', type=float, help='x boundary. Example: 0.0 1.0')
    parser.add_argument('--y_bounds', nargs='+', type=float, help='y boundary. Example: 0.0 1.0')
    parser.add_argument('--dx', default=0.025, type=float, help="Spacing for meshing")
    parser.add_argument('--dy', default=0.025, type=float, help="Spacing for meshing")
    # Inputs for particles, particle_sets
    parser.add_argument('--nparticle_per_dir', default=4, help="Number of particles to generate in mesh in one direction")
    parser.add_argument('--x_range', nargs='+', type=float, help='Rectangular sand x range. Example: 0.0 1.0')
    parser.add_argument('--y_range', nargs='+', type=float, help='Rectangular sand y range. Example: 0.0 1.0')
    parser.add_argument('--randomness', default=0.9, type=float, help="Random uniform distribution of particle arrangement")
    # Inputs for particle stress
    parser.add_argument('--k0', type=float, help="K0 for geostatic")
    parser.add_argument('--density', type=float, help="Density of soil")
    # Inputs for initial velocity of particles
    parser.add_argument('--initial_vel', nargs='+', type=float, help="Initial velocity value just for plotting configuration figure. Example: 0.1 0.2")
    args = parser.parse_args()

    # Inputs for mesh, node, node_sets
    ndim = args.ndim
    x_bounds = args.x_bounds
    y_bounds = args.y_bounds
    dx = args.dx
    dy = args.dy
    nnode_in_ele= 4
    # Inputs for particles, particle_sets
    nparticle_per_dir = args.nparticle_per_dir
    x_range = args.x_range
    y_range = args.y_range
    randomness = args.randomness
    # Inputs for particle stress
    k0 = args.k0
    density = args.density
    if density is not None:
        unit_weight = density * 9.81
    else:
        pass
    # Inputs for initial velocity of particles
    initial_vel = args.initial_vel

    #%% Mesh, node

    # Calculate number of nodes and elements
    nnode_x = (x_bounds[1] - x_bounds[0]) / dx + 1
    nnode_y = (y_bounds[1] - y_bounds[0]) / dy + 1
    nele_x = nnode_x - 1
    nele_y = nnode_y - 1
    nnode = nnode_x * nnode_y
    nele = nele_x * nele_y

    # Generate mesh node coordinates
    xs = np.arange(x_bounds[0], x_bounds[1] + dx, dx)
    ys = np.arange(y_bounds[0], y_bounds[1] + dy, dy)
    xy = []
    for y in ys:
        for x in xs:
            xy.append([x, y])
    xy = np.array(xy)

    # Make cell groups
    cells = np.empty((int(nele), int(nnode_in_ele)))
    i = 0
    for ely in range(int(nele_y)):
        for elx in range(int(nele_x)):
            # cell index starts from 1 not 0, so there is "1+" at first
            cells[i, 0] = nnode_x * ely + elx
            cells[i, 1] = nnode_x * ely + elx + 1
            cells[i, 2] = nnode_x * (ely + 1) + elx + 1
            cells[i, 3] = nnode_x * (ely + 1) + elx
            i += 1
    cells = cells.astype(int)

    print("Make `mesh.txt`")
    # Write the number of nodes
    f = open("mesh.txt", "w")
    f.write(f"{int(nnode)}\t{int(nele)}\n")
    f.close()

    # Append coordinate values of nodes to 'mesh.txt'
    f = open('mesh.txt', 'a')
    f.write(
        np.array2string(
            # xy, formatter={'float_kind':lambda lam: "%.4f" % lam}, separator='\t', threshold=math.inf
            xy, separator='\t', threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.write('\n')
    f.close()

    # Append cell groups to 'mesh.txt'
    f = open('mesh.txt', 'a')
    f.write(
        np.array2string(
            # cells, formatter={'float_kind':lambda lam: "%.4f" % lam}, separator='\t', threshold=math.inf
            cells, separator='\t', threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.close()


    #%% Particles

    def create_particle_array(nparticle_per_dir, x_bound, y_bound, dx, dy, ndim=2, randomness=0.8):
        # Geometry
        offset = dx / nparticle_per_dir / 2
        particle_interval = dx / nparticle_per_dir
        xmin = x_bound[0] + offset
        xmax = x_bound[1] - offset
        ymin = y_bound[0] + offset
        ymax = y_bound[1] - offset

        # Create particle arrays
        xs = np.arange(xmin, xmax + offset, particle_interval)
        ys = np.arange(ymin, ymax + offset, particle_interval)
        xy = []
        for y in ys:
            for x in xs:
                xy.append([x, y])
        xy = np.array(xy)
        xy = xy + np.random.uniform(-offset*randomness, offset*randomness, size=xy.shape)
        return xy

    particle_set1 = create_particle_array(
        nparticle_per_dir=nparticle_per_dir,
        x_bound=x_range, y_bound=y_range,
        dx=dx, dy=dy,
        randomness=randomness
    )
    particles = particle_set1
    nparticles = particles.shape[0]

    print("Make `particles.txt`")
    # Write the number of particles
    f = open("particles.txt", "w")
    f.write(f"{nparticles} \n")
    f.close()

    # Write coordinates for particles
    f = open("particles.txt", "a")
    f.write(
        np.array2string(
            # particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
            particles, threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.close()
    
    # Particle stresses
    if k0 is not None:
        print(f"Make `particles_stresses.txt` with K0={k0}, density={density}")
        particle_stress = np.zeros((np.shape(particles)[0], 3))  # second axis is for stress xx, yy, zz
        particle_stress[:, 0] = k0 * (y_range[1] - particles[:, 1]) * unit_weight  # K0*H*Unit_Weight
        particle_stress[:, 1] = (y_range[1] - particles[:, 1]) * unit_weight  # H*Unit_Weight
        particle_stress[:, 2] = 0  # for 2d case stress zz is zero

        # Write the number of stressed particles
        f = open("particles-stresses.txt", "w")
        f.write(f"{np.shape(particles)[0]} \n")
        f.close()

        # Write coordinates for particles
        f = open("particles-stresses.txt", "a")
        f.write(
            np.array2string(
                # particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
                particle_stress, threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
        )
        f.close()
    else:
        print(f"K0 not provided. Skip making particles_stresses.txt")

    # plot
    fig, ax = plt.subplots()
    ax.scatter(particles[:, 0], particles[:, 1], s=0.5)
    if initial_vel is not None:
        # show velocity quiver and value
        x_center = (particles[:, 0].max() - particles[:, 0].min())/2 + particles[:, 0].min()
        y_center = (particles[:, 1].max() - particles[:, 1].min())/2 + particles[:, 1].min()
        ax.quiver(x_center, y_center, initial_vel[0], initial_vel[1], scale=10)
        ax.text(x_center, y_center, f"vel = {str(initial_vel)}")
    # show granular group box location
    ax.text(x_range[0], y_range[0], f"[{x_range[0]}, {y_range[0]}]")
    ax.text(x_range[0], y_range[1], f"[{x_range[0]}, {y_range[1]}]")
    ax.text(x_range[1], y_range[0], f"[{x_range[1]}, {y_range[0]}]")
    ax.text(x_range[1], y_range[1], f"[{x_range[1]}, {y_range[1]}]")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')
    ax.set_title(f"Cell size={dx}x{dy}, Particle/cell={nnode_in_ele**2}, nparticles={nparticles} \n"
                 f"Particle coordinates: "
                 f"[{x_range[0]}, {y_range[0]}],"
                 f"[{x_range[0]}, {y_range[1]}],"
                 f"[{x_range[1]}, {y_range[0]}],"
                 f"[{x_range[1]}, {y_range[1]}]", fontsize=10)
    plt.savefig('initial_config.png')
   

    #%% Entities
    #
    # x_bounds_node_id = []
    # y_bounds_node_id = []
    # particles_id = []
    #
    # # Find index of nodes that match boundaries
    # for i, coordinate in enumerate(xy):
    #     # x boundaries
    #     if coordinate[0] == x_bounds[0] or coordinate[0] == x_bounds[1]:
    #         x_bounds_node_id.append(i)
    #     if coordinate[1] == y_bounds[0] or coordinate[1] == y_bounds[1]:
    #         y_bounds_node_id.append(i)
    # for i, coordinate in enumerate(particle_set1):
    #     if (x_range[0] <= coordinate[0] <= x_range[1]) \
    #         and (y_range[0] <= coordinate[1] <= y_range[1]):
    #         particles_id.append(i)

    left_bound_node_id = []
    right_bound_node_id = []
    bottom_bound_node_id = []
    upper_bound_node_id = []
    particles_id = []

    # Find index of nodes that match boundaries
    for i, coordinate in enumerate(xy):
        if coordinate[0] == x_bounds[0]:
            left_bound_node_id.append(i)
        if coordinate[0] == x_bounds[1]:
            right_bound_node_id.append(i)
        if coordinate[1] == y_bounds[0]:
            bottom_bound_node_id.append(i)
        if coordinate[1] == y_bounds[1]:
            upper_bound_node_id.append(i)
    for i, coordinate in enumerate(particle_set1):
        if (x_range[0] <= coordinate[0] <= x_range[1]) \
            and (y_range[0] <= coordinate[1] <= y_range[1]):
            particles_id.append(i)

    #%% Write `entity_sets.json`
    entity_sets = {
        "node_sets": [
            {
                "id": 0,
                "set": bottom_bound_node_id
            },
            {
                "id": 1,
                "set": upper_bound_node_id
            },
            {
                "id": 2,
                "set": left_bound_node_id
            },
            {
                "id": 3,
                "set": right_bound_node_id
            }
        ],
        "particle_sets": [
                {
                    "id": 0,
                    "set": particles_id
                }
            ]
    }

    print("Make `entity_sets.json`")
    with open("entity_sets.json", "w") as f:
        json.dump(entity_sets, f, indent=2)
    f.close()
