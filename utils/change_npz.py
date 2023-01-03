import numpy as np
import json

data = np.load(f'gns-data/datasets/droplet2/merged_train3d.npz', allow_pickle=True)
save_name = "merged_train2d"
original_sequence_length = 25001
sequence_downsample_rate = 100
trajectories = {}
running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
aggregated_positions = []
aggregated_velocities = []
aggregated_accelerations = []
data_names = None

bounds = [[-1.2418140152084631, 46.841814015208456], [-5.16455663821927, 9.321478576384568]]
sequence_length = int(original_sequence_length/sequence_downsample_rate) + 1
default_connectivity_radius = 10.0
dim = int(2)
dt_mpm = 0.01  # 0.0025
mpm_cell_size = None  # [0.0125, 0.0125]
nparticles_per_cell = None  # int(16)
dt_gns = dt_mpm

# assume that `trajectory` in `data.items()` has already changed from 3d to 2d.
for i, (simulation_id, (trajectory3d, material)) in enumerate(data.items()):
    # print(i)
    # trajectory[0] is position sequence (nsequence, nparticles, ndim)
    # trajectory[1] is material type integer (nparticles)
    trajectory2d = trajectory3d[::sequence_downsample_rate, :, [0, 2]]
    trajectories[simulation_id] = (trajectory2d, material)

    # get positions for each mpm simulation
    array_shape = trajectory2d.shape
    flattened_positions = np.reshape(trajectory2d, (-1, array_shape[-1]))

    # compute velocities using finite difference
    # assume velocities before zero are equal to zero
    velocities = np.empty_like(trajectory2d)
    velocities[1:] = (trajectory2d[1:] - trajectory2d[:-1]) / dt_gns
    velocities[0] = 0
    flattened_velocities = np.reshape(velocities, (-1, array_shape[-1]))

    # compute accelerations finite difference
    # assume accelerations before zero are equal to zero
    accelerations = np.empty_like(velocities)
    accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt_gns
    accelerations[0] = 0
    flattened_accelerations = np.reshape(accelerations, (-1, array_shape[-1]))

    # aggregate the arrays
    aggregated_positions.append(flattened_positions)
    aggregated_velocities.append(flattened_velocities)
    aggregated_accelerations.append(flattened_accelerations)

    # # Save npz
    # np.savez_compressed(f"{save_name}-test{i}.npz", **trajectories)
    # print(f"npz saved at: ./{save_name}-test{i}.npz")

# Concatenate the aggregated arrays
concat_positions = np.concatenate(aggregated_positions)
concat_velocities = np.concatenate(aggregated_velocities)
concat_accelerations = np.concatenate(aggregated_accelerations)

# Compute statistics
statistics = {
    "mean_velocity_x": float(np.mean(concat_velocities[:, 0])),
    "mean_velocity_y": float(np.mean(concat_velocities[:, 1])),
    "std_velocity_x": float(np.std(concat_velocities[:, 0])),
    "std_velocity_y": float(np.std(concat_velocities[:, 1])),
    "mean_accel_x": float(np.mean(concat_accelerations[:, 0])),
    "mean_accel_y": float(np.mean(concat_accelerations[:, 1])),
    "std_accel_x": float(np.std(concat_accelerations[:, 0])),
    "std_accel_y": float(np.std(concat_accelerations[:, 1]))
}
print(statistics)
# Save npz
np.savez_compressed(f"{save_name}.npz", **trajectories)
print(f"npz saved at: ./{save_name}.npz")

# Save metadata.json
metadata = {
    "bounds": bounds,
    "sequence_length": sequence_length,
    "default_connectivity_radius": float(default_connectivity_radius),
    "dim": dim,
    "dt": float(dt_mpm),
    "vel_mean": [statistics["mean_velocity_x"], statistics["mean_velocity_y"]],
    "vel_std": [statistics["std_velocity_x"], statistics["std_velocity_y"]],
    "acc_mean": [statistics["mean_accel_x"], statistics["mean_accel_y"]],
    "acc_std": [statistics["std_accel_x"], statistics["std_accel_y"]]
}

with open(f"metadata-{save_name}.json", "w") as fp:
    json.dump(metadata, fp)
print(f"metadata saved at: ./{save_name}.json")