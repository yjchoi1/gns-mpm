import torch
import torch_geometric
from torch_geometric.nn import radius_graph
from gns import data_loader
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

#%% See train.npz
data_path = "../gns-data/datasets/sand-2d/train.npz"
npz_data = np.load(data_path, allow_pickle=True)
for id, trajectory in npz_data.items():
    print(id)
    print(trajectory[0].shape)
    print(trajectory[1].shape)


#%% Torch radius

node_features = torch.tensor([[0, 0],
                              [1, 0],
                              [1, 1],
                              [0, 1]])
edge_index = radius_graph(
    node_features, r=2, batch=None)



#%% See ds to verify what is batch


data_path = "../gns-data/datasets/sand-2d/"
ds = data_loader.get_data_loader_by_samples(path=f"{data_path}train.npz",
                                              input_length_sequence=6,
                                              batch_size=2,
                                              shuffle=False)

# length of ds is, (trajectory_length - 6)*(n_trajectory)/2
print(len(ds)) # 1413 =  (320-6)*9/2

# Look at the first 2 trajectory.
for i, ((position, particle_type, n_particles_per_example), labels) in enumerate(ds):
    if i == 1:
        break
    print(position.shape)  # torch.Size([12800, 6, 2])
    print(particle_type.shape)  # torch.Size([12800])
    print(n_particles_per_example.shape)  # torch.Size([2])
    print(n_particles_per_example)  # tensor([6400, 6400])
    print(labels.shape)  # torch.Size([12800, 2])
    print(i)



#%% _compute_graph_connectivity

# `data_loader` imports `ds` which is a tuple:

#     (
#        (print(position.shape)  # torch.Size([12800, 6, 2]),
#         print(particle_type.shape)  # torch.Size([12800]),
#         print(n_particles_per_example.shape)  # torch.Size([2]),
#         print(n_particles_per_example)  # tensor([6400, 6400]),
#         ),
#        (
#         print(labels.shape)  # torch.Size([12800, 2]
#        )
#     )

# As you can see, it is concatenated list of two batch
# Let's make try the similar array
node_features = [
    np.array([[x, y] for x in np.linspace(0, 1, 2) for y in np.linspace(0, 1, 2)]),
    np.array([[x, y] for x in np.linspace(1.5, 2.5, 2) for y in np.linspace(1.5, 2.5, 2)])
]
nparticles_per_example = [len(node_features[0]), len(node_features[1])]
node_features = np.concatenate(node_features)
radius = 1.5

# Batch_ids are the flattened ids (0 or 1) represents
# where the node features are come from which batch.
batch_ids = torch.cat(
    [torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(nparticles_per_example)])

# radius_graph accepts r < radius not r <= radius
# A torch tensor list of source and target nodes with shape (2, nedges)
edge_index = radius_graph(
    torch.tensor(node_features), r=radius, batch=batch_ids, loop=False)
# edge_index = np.array(edge_index)

# sender and receiver
receiver = edge_index[0, :]
sender = edge_index[1, :]

# Find edge index where sender==receiver
edge_index_inverted = torch.empty(edge_index.shape, dtype=torch.int64)
edge_index_inverted[[0, 1], :] = edge_index[[1, 0], :]

bidirectional_edge_index = torch.tensor(
    [[i, j+i] for i, sender2receiver in enumerate(edge_index.T)
            for j, receiver2sender in enumerate(edge_index_inverted.T[i:])
            if torch.equal(sender2receiver, receiver2sender)]
)
print(bidirectional_edge_index)

# for i, sender2receiver in enumerate(edge_index.T):
#     for j, receiver2sender in enumerate(edge_index_inverted.T[i:]):
#         if torch.equal(sender2receiver, receiver2sender):
#             print(i, j+i)


# Show graph connectivity plot
a = torch_geometric.data.Data(x=torch.tensor(node_features), edge_index=torch.tensor(edge_index))
g = torch_geometric.utils.to_networkx(a, to_undirected=True)
nx.draw(g)
plt.savefig("filename.png")

#%% Draw graph
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = torch_geometric.data.Data(x=x, edge_index=edge_index)
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
# plt.figure()
nx.draw(g)
plt.savefig("filename.png")

