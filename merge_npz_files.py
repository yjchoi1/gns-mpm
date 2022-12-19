import numpy as np
import json

# Inputs
bounds = [[-1.0225332856518696, -1.0, -1.0370372322163504], [46.62253328565188, 1.0, 5.1939591703816586]]
sequence_length = int(20)
default_connectivity_radius = 10.0
dim = int(3)
dt_mpm = None  # 0.0025
mpm_cell_size = None  # [0.0125, 0.0125]
nparticles_per_cell = None  # int(16)
dt_gns = 0.01

mpm_dir = "./gns-data/datasets/test_files" # "./mpm"
data_case = "test" # "mpm-9k-train"
# data_tags = [str(i) for i in np.arange(0, 24)] + [str(25), str(26)]
data_tags = np.arange(1, 21, 1)
save_name = "train-test"


trajectories = {}
running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
aggregated_positions = []
aggregated_velocities = []
aggregated_accelerations = []
data_names = []

for id in data_tags:
    data_name = f"{data_case}{id}"
    data_names.append(data_name)
    npz_path = f"{mpm_dir}/{data_name}.npz"  # f"{mpm_dir}/{data_name}/{data_name}.npz"
    data = np.load(npz_path, allow_pickle=True)
    for simulation_id, trajectory in data.items():
        trajectories[f"simulation_trajectory_{id}"] = (trajectory)

        # get positions for each mpm simulation
        positions = trajectory[0]
        array_shape = positions.shape
        flattened_positions = np.reshape(positions, (-1, array_shape[-1]))

        # compute velocities using finite difference
        # assume velocities before zero are equal to zero
        velocities = np.empty_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / dt_gns
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
    print(f"{key}: {value:.7E}")

# Save npz
np.savez_compressed(f"{save_name}.npz", **trajectories)
print(f"npz saved at: ./{save_name}.npz")

# Save metadata.json
metadata = {
    "bounds": bounds,
    "sequence_length": sequence_length,
    "default_connectivity_radius": default_connectivity_radius,
    "dim": dim,
    "dt": dt_mpm,
    "vel_mean": [statistics["mean_velocity_x"], statistics["mean_velocity_y"]],
    "vel_std": [statistics["std_velocity_x"], statistics["std_velocity_y"]],
    "acc_mean": [statistics["mean_accel_x"], statistics["mean_accel_y"]],
    "acc_std": [statistics["std_accel_x"], statistics["std_accel_y"]],
    "mpm_cell_size": mpm_cell_size,
    "nparticles_per_cell": nparticles_per_cell,
    "data_names": data_names
}

with open(f"metadata-{save_name}.json", "w") as fp:
    json.dump(metadata, fp)
print(f"metadata saved at: ./{save_name}.json")


# See npz
data = np.load('train.npz', allow_pickle=True)
for simulation_id, trajectory in data.items():
    print(simulation_id)
    print(trajectory)
