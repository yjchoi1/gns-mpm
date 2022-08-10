import torch
import torch_geometric
from torch_geometric.nn import radius_graph
# from gns import data_loader
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import pickle

#%%
pkl_file = open('edge_features.pkl', 'rb')
edge_features = np.array(pickle.load(pkl_file))

pkl_file = open('edge_index.pkl', 'rb')
edge_index = np.array(pickle.load(pkl_file))

pkl_file = open('connectivity_indices.pkl', 'rb')
connectivity_indices = pickle.load(pkl_file)

pkl_file.close()

message0 = edge_features[0]  # node 5 and 0
message49 = edge_features[49]  # node 0 and 5
message127 = edge_features[127]  # node 26 and 13
message256 = edge_features[256]  # node 13 and 26
diff0and49 = message0 - message49
diff127and256 = message127 - message256
diff0and127 = message0 - message127

fig1, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(np.arange(128), message0,
        label="message0 (node 5-0)", marker=">", color="black")
ax.plot(np.arange(128), message49,
        label="message49 (node 0-5)", marker="<", color="gray")
ax.plot(np.arange(128), message127,
        label="message127 (node 26-13)", marker=">", color="red")
ax.plot(np.arange(128), message256,
        label="message256 (node 13-26)", marker="<", color="tomato")
ax.set_xlabel("feature_id")
ax.set_ylabel("value")
# ax.set_xlim([0, 60])
# ax.set_ylim([-1.1, 1.1])
ax.legend(prop={'size': 6})
ax.grid()
# ax.set_title("message values for edge 0 and 127")
# ax.set_title("message values for edge 49 and 256")
plt.tight_layout()
fig1.show()

fig1, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax.plot(np.arange(128), message0,
        label="message0 (node 5-0)", marker=">", color="black")
ax.plot(np.arange(128), -message49,
        label="message49 (node 0-5)", marker="<", color="gray")
ax.set_xlabel("feature_id")
ax.set_ylabel("value")
ax.set_xlim([0, 60])
ax.set_ylim([-1.1, 1.1])
plt.tight_layout()
fig1.show()

fig2, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(128), abs(message0 - message49), label="abs(message0-49)")
ax[0].plot(np.arange(128), abs(message127 - message256), label="abs(message127-256)")
ax[1].plot(np.arange(128), abs(message0 - message127), label="abs(message0-127)")
ax[1].plot(np.arange(128), abs(message49 - message256), label="abs(message49-256)")
for a in ax:
    a.set_xlabel("feature_id")
    a.set_ylabel("value")
    # a.set_title("message diffs")
    a.set_xlim([0, 60])
    a.legend(prop={'size': 6})
    # ax.legend(prop={'size': 6})
ax[0].set_ylim([-1.1, 1.1])
ax[1].set_ylim([-0.5, 0.5])
ax[0].set_title("Edges connecting the same nodes pair")
ax[1].set_title("Edges connecting different nodes pair")
plt.tight_layout()
fig2.show()

# ax[1].plot(np.arange(128), diff_0and100, label="0to49 - 49to0")
# ax[1].plot(np.arange(128), diff_0and49, label="0to49 - 6to10")
# ax[1].set_xlabel("feature_id")
# ax[1].set_ylabel("value")
# ax[1].set_xlim(0, 128)
# ax[1].set_title("message difference")
# ax[1].legend()


fig2, ax2 = plt.subplots()
ax2.plot(np.arange(128), message0, label="message_0to49")
ax2.plot(np.arange(128), negative_message49, label="negative message_49to0")
ax2.set_xlabel("feature_id")
ax2.set_ylabel("value")
ax2.set_title("message values")
ax2.legend()
fig2.show()






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

