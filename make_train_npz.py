import numpy as np

# Inputs
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
save_name = "train.npz"
dt = 1.0

trajectories = {}
running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
aggregated_positions = []
aggregated_velocities = []
aggregated_accelerations = []

for id in ids:
    data = np.load(f'mpm/mpm-train{id}/train-{id}.npz', allow_pickle=True)
    for simulation_id, trajectory in data.items():
        trajectories[f"simulation_trajectory_{id}"] = (trajectory)

        # get positions for each mpm simulation
        positions = trajectory[0]
        array_shape = positions.shape
        flattened_positions = np.reshape(positions, (-1, array_shape[-1]))

        # compute velocities using finite difference
        # assume velocities before zero are equal to zero
        velocities = np.empty_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / dt
        velocities[0] = 0
        flattened_velocities = np.reshape(velocities, (-1, array_shape[-1]))

        # compute accelerations finite difference
        # assume accelerations before zero are equal to zero
        accelerations = np.empty_like(velocities)
        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
        accelerations[0] = 0
        flattened_accelerations = np.reshape(accelerations, (-1, array_shape[-1]))

        # aggregate the arrays
        aggregated_positions.append(flattened_positions)
        aggregated_velocities.append(flattened_velocities)
        aggregated_accelerations.append(flattened_accelerations)

# Concatenate the aggregated arrays
concat_positions = np.concatenate(aggregated_positions)
concat_velocities = np.concatenate(aggregated_velocities)
concat_accelerations = np.concatenate(aggregated_accelerations)

# Compute statistics
statistics = {
    "mean_velocity_x": np.mean(concat_velocities[:, 0]),
    "mean_velocity_y": np.mean(concat_velocities[:, 1]),
    "std_velocity_x": np.std(concat_velocities[:, 0]),
    "std_velocity_y": np.std(concat_velocities[:, 1]),
    "mean_accel_x": np.mean(concat_accelerations[:, 0]),
    "mean_accel_y": np.mean(concat_accelerations[:, 1]),
    "std_accel_x": np.std(concat_accelerations[:, 0]),
    "std_accel_y": np.std(concat_accelerations[:, 1])
}

# Print statistics
for key, value in statistics.items():
    print(f"{key}: {value:.5E}")

# Save npz
np.savez_compressed(save_name, **trajectories)

# # See npz
# data = np.load('train.npz', allow_pickle=True)
# for simulation_id, trajectory in data.items():
#     print(simulation_id)
#     print(trajectory)
