import numpy as np
import math
import json
import argparse

if __name__ == "__main__":
    #%% Inputs for mesh, node, node_sets
    parser = argparse.ArgumentParser(description='Make mesh and particle inputs to run mpm')
    parser.add_argument('--ndim', default=2, help="Dimension of simulation domain.")
    parser.add_argument('--x_bounds', nargs='+', type=float, help='x boundary. Example: 0.0 1.0')
    parser.add_argument('--y_bounds', nargs='+', type=float, help='y boundary. Example: 0.0 1.0')
    parser.add_argument('--dx', default=0.025, help="Spacing for meshing")
    parser.add_argument('--dy', default=0.025, help="Spacing for meshing")
    # Inputs for particles, particle_sets
    parser.add_argument('--nparticle_per_dir', default=4, help="Number of particles to generate in mesh in one direction")
    parser.add_argument('--x_range', nargs='+', type=float, help='Rectangular sand x range. Example: 0.0 1.0')
    parser.add_argument('--y_range', nargs='+', type=float, help='Rectangular sand y range. Example: 0.0 1.0')
    parser.add_argument('--randomness', default=0.9, help="Random uniform distribution of particle arrangement")
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
    xy = np.round(xy, 5)

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

    # Write the number of nodes
    f = open("mesh.txt", "w")
    f.write(f"{int(nnode)}\t{int(nele)}\n")
    f.close()

    # Append coordinate values of nodes to 'mesh.txt'
    f = open('mesh.txt', 'a')
    f.write(
        np.array2string(
            xy, formatter={'float_kind':lambda lam: "%.4f" % lam}, separator='\t', threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.write('\n')
    f.close()

    # Append cell groups to 'mesh.txt'
    f = open('mesh.txt', 'a')
    f.write(
        np.array2string(
            cells, formatter={'float_kind':lambda lam: "%.4f" % lam}, separator='\t', threshold=math.inf
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
        xy = np.round(xy, 5)
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

    # Write the number of particles
    f = open("particles.txt", "w")
    f.write(f"{nparticles} \n")
    f.close()

    # Write coordinates for particles
    f = open("particles.txt", "a")
    f.write(
        np.array2string(
            particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.close()


    #%% Entities

    x_bounds_node_id = []
    y_bounds_node_id = []
    particles_id = []

    # Find index of nodes that match boundaries
    for i, coordinate in enumerate(xy):
        # x boundaries
        if coordinate[0] == x_bounds[0] or coordinate[0] == x_bounds[1]:
            x_bounds_node_id.append(i)
        if coordinate[1] == y_bounds[0] or coordinate[1] == y_bounds[1]:
            y_bounds_node_id.append(i)
    for i, coordinate in enumerate(particle_set1):
        if (x_range[0] <= coordinate[0] <= x_range[1]) \
            and (y_range[0] <= coordinate[1] <= y_range[1]):
            particles_id.append(i)


    #%% Write `entity_sets.json`

    entity_sets = {
        "node_sets": [
            {
                "id": 0,
                "set": x_bounds_node_id
            },
            {
                "id": 1,
                "set": y_bounds_node_id
            }
            # {
            #     "id": 2,
            #     "set": particles
            # }
        ],
        "particle_sets": [
                {
                    "id": 0,
                    "set": particles_id
                }
            ]
    }

    with open("entity_sets.json", "w") as f:
        json.dump(entity_sets, f, indent=2)
