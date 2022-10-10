import numpy as np
import math
import json

# Inputs
ndim = 3
xmin = 0.1
ymin = 0.1
zmin = 0.1
xmax = 0.9
ymax = 0.9
zmax = 0.9
dx = 0.01
dy = 0.01
dz = 0.01
nnode_in_ele= 8

# Calculate number of nodes and elements
nnode_x = (xmax - xmin)/dx + 1
nnode_y = (ymax - ymin)/dy + 1
nnode_z = (zmax - zmin)/dz + 1
nnode = nnode_x*nnode_y*nnode_z
nele_x = nnode_x - 1
nele_y = nnode_y - 1
nele_z = nnode_z - 1
nele = nele_x*nele_y*nele_z

# Generate mesh node coordinates
x = np.arange(xmin, xmax + dx, dx)
y = np.arange(ymin, ymax + dy, dy)
z = np.arange(zmin, zmax + dy, dz)
xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, ndim)
xyz = xyz[:, [1, 0, 2]]
xyz = np.round(xyz, 5)

# Make cell groups
cells = np.empty((int(nele), int(nnode_in_ele)))
i = 0
for elz in range(int(nele_z)):
    for ely in range(int(nele_y)):
        for elx in range(int(nele_x)):
            # cell index starts from 1 not 0, so there is "1+" at first
            cells[i, 0] = nnode_x*nnode_y*elz + ely*nnode_x + elx
            cells[i, 1] = nnode_x*nnode_y*elz + ely*nnode_x + 1 + elx
            cells[i, 2] = nnode_x*nnode_y*elz + (ely+1)*nnode_x + 1 + elx
            cells[i, 3] = nnode_x*nnode_y*elz + (ely+1)*nnode_x + elx
            cells[i, 4] = nnode_x*nnode_y*(elz+1) + ely*nnode_x + elx
            cells[i, 5] = nnode_x*nnode_y*(elz+1) + ely*nnode_x + 1 + elx
            cells[i, 6] = nnode_x*nnode_y*(elz+1) + (ely+1)*nnode_x + 1 + elx
            cells[i, 7] = nnode_x*nnode_y*(elz+1) + (ely+1)*nnode_x + elx
            i += 1

cells = cells.astype(int)

# Write the number of nodes
f = open("mesh.txt", "w")
f.write(f"{nnode}\t{nele}\n")
f.close()

# Append coordinate values of nodes to 'mesh.txt'
f = open('mesh.txt', 'a')
f.write(np.array2string(
    xyz, separator='\t', threshold=math.inf).replace(' [', '').replace('[', '').replace(']', ''))
f.write('\n')
f.close()

# Append cell groups to 'mesh.txt'
f = open('mesh.txt', 'a')
f.write(np.array2string(
    cells, separator='\t', threshold=math.inf).replace(' [', '').replace('[', '').replace(']', ''))
f.close()

#%% Entities

id0 = []
id1 = []
id2 = []

# Find index of nodes that match boundaries
for i, coordinate in enumerate(xyz):
    # x boundaries
    if coordinate[0] == xmin or coordinate[0] == xmax:
        id0.append(i)
    # y boundaries
    if coordinate[1] == ymin or coordinate[1] == ymax:
        id1.append(i)
    # z boundaries
    if coordinate[2] == zmin:
        id2.append(i)

# Write `entity_sets.json`
entity_sets = {
    "node_sets": [
        {
            "id": 0,
            "set": id0
        },
        {
            "id": 1,
            "set": id1
        },
        {
            "id": 2,
            "set": id2
        },
    ]
}

with open("entity_sets.json", "w") as f:
    json.dump(entity_sets, f, indent=2)

#%%
