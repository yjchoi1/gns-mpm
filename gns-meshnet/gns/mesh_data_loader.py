import torch
import numpy as np

class SamplesDataset(torch.utils.data.Dataset):

    def __init__(self, path, input_length_sequence):
        super().__init__()
        # load dataset stored in npz format.
        # data consists of dict with keys:
        # ["pos", "node_type", "velocity", "cells", "pressure"] for all trajectory.
        # whose shapes are (600, 1876, 2), (600, 1876, 1), (600, 1876, 2), (600, 3518, 3), (600, 1876, 1)
        # convert to list of tuples
        self._data = [dict(trj_info.item()) for trj_info in np.load(path, allow_pickle=True).values()]

        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["pos"].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [x["pos"].shape[0] - input_length_sequence for x in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # # Prepare training data.
        # positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        # positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        # particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        # n_particles_per_example = positions.shape[0]
        # label = self._data[trajectory_idx][0][time_idx]
        # training_example = ((positions, particle_type, n_particles_per_example), label)

        # Prepare training data. Assume `input_sequence_length`=1
        positions = self._data[trajectory_idx]["pos"][time_idx - 1]  # (nnode, dimension)
        # positions = np.transpose(positions)
        n_node_per_example = positions.shape[1]  # (nnode, )
        node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)
        velocity_feature = self._data[trajectory_idx]["velocity"][time_idx - 1]  # (nnode, dimension)
        velocity_target = self._data[trajectory_idx]["velocity"][time_idx]  # (nnode, dimension)
        pressure = self._data[trajectory_idx]["pressure"][time_idx - 1]  # (nnode, 1)
        cells = self._data[trajectory_idx]["cells"][time_idx - 1]  # (ncells, nnode_per_cell)
        time_idx_vector = np.full(positions.shape[0], time_idx)  # (nnode, )

        training_example = (
            (positions, node_type, velocity_feature, pressure, cells, time_idx_vector, n_node_per_example),
            velocity_target
        )

        return training_example

def collate_fn(data):
    position_list = []
    node_type_list = []
    velocity_feature_list = []
    pressure_list = []
    cell_list_list = []
    time_idx_vector_list = []
    n_node_per_example_list = []
    velocity_target_list = []

    for (feature, label) in data:
        position_list.append(feature[0])  # (nnode, input_sequence_length, dimension)
        node_type_list.append(feature[1])  # (nnode, )
        velocity_feature_list.append(feature[2])  # (
        pressure_list.append(feature[3])
        cell_list_list.append(feature[4])
        time_idx_vector_list.append(feature[5])
        n_node_per_example_list.append(feature[6])
        velocity_target_list.append(label)

    collated_data = (
        (
            torch.tensor(np.concatenate(position_list)).to(torch.float32).contiguous(),
            torch.tensor(np.concatenate(node_type_list)).contiguous(),
            torch.tensor(np.concatenate(velocity_feature_list)).to(torch.float32).contiguous(),
            torch.tensor(np.concatenate(pressure_list)).to(torch.float32).contiguous(),
            torch.tensor(np.concatenate(cell_list_list)).contiguous(),
            torch.tensor(np.concatenate(time_idx_vector_list)).to(torch.float32).contiguous(),
            torch.tensor(n_node_per_example_list).contiguous(),
        ),
        torch.tensor(np.concatenate(velocity_target_list)).to(torch.float32).contiguous(),
    )

    return collated_data

def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)