import numpy as np
import math
import json

# Inputs
ndim = 2
xmin = 0.0
ymin = 0.0
xmax = 1.0
ymax = 1.0
dx = 0.010
dy = 0.010
nnode_in_ele= 4
particle_geometry = [[0.2, 0.3], [0.1, 0.5]]

# Calculate number of nodes and elements
nnode_x = (xmax - xmin)/dx + 1
nnode_y = (ymax - ymin)/dy + 1
nele_x = nnode_x - 1
nele_y = nnode_y - 1
nnode = nnode_x * nnode_y
nele = nele_x * nele_y

# Generate mesh node coordinates
xs = np.arange(xmin, xmax + dx, dx)
ys = np.arange(ymin, ymax + dy, dy)
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

# Entities

x_bounds = []
y_bounds = []
particles = []

# Find index of nodes that match boundaries
for i, coordinate in enumerate(xy):
    # x boundaries
    if coordinate[0] == xmin or coordinate[0] == xmax:
        x_bounds.append(i)
    if coordinate[1] == ymin or coordinate[1] == ymax:
        y_bounds.append(i)
    if (particle_geometry[0][0] <= coordinate[0] <= particle_geometry[0][1]) \
        and (particle_geometry[1][0] <= coordinate[1] <= particle_geometry[1][1]):
        particles.append(i)

# Write `entity_sets.json`
entity_sets = {
    "node_sets": [
        {
            "id": 0,
            "set": x_bounds
        },
        {
            "id": 1,
            "set": y_bounds
        },
        {
            "id": 2,
            "set": particles
        }
    ]
}

with open("entity_sets.json", "w") as f:
    json.dump(entity_sets, f, indent=2)
