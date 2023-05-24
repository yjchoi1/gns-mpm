import numpy as np
import json

# Inputs
bounds = [[-0.020833333333333332, 1.5208333333333333], [-0.2708333332, 0.2708333332], [-0.020833333333333332, 1.0208333333333333]]
sequence_length = int(380)
default_connectivity_radius = 0.041
dim = int(3)
material_feature_len = int(0)
dt_mpm = 0.0025  # 0.0025
mpm_cell_size = [1/12, 1/12, 1/12]  # [0.0125, 0.0125]
nparticles_per_cell = int(4*4)  # int(16)
dt_gns = 1.0  # 1.0 is default

mpm_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d/"  # "./mpm"
data_case = "sand3d_column_collapse"  # "mpm-9k-train"
data_tags = [i for i in range(8, 16)]
excluded_data_tags = [83, 144, 148]
data_tags = [i for i in data_tags if i not in excluded_data_tags]
save_name = "sand3d_column_val"


trajectories = {}
if dim == 2:
    running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
if dim == 3:
    running_sum = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
    running_sumsq = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
    running_count = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
aggregated_positions = []
aggregated_velocities = []
aggregated_accelerations = []
data_names = []

for id in data_tags:
    data_name = f"{data_case}{id}"
    data_names.append(data_name)
    npz_path = f"{mpm_dir}/{data_name}/{data_name}.npz"  # f"{mpm_dir}/{data_name}/{data_name}.npz"
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
if dim == 2:
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
if dim == 3:
    statistics = {
        "mean_velocity_x": np.mean(concat_velocities[:, 0]),
        "mean_velocity_y": np.mean(concat_velocities[:, 1]),
        "mean_velocity_z": np.mean(concat_velocities[:, 2]),
        "std_velocity_x": np.std(concat_velocities[:, 0]),
        "std_velocity_y": np.std(concat_velocities[:, 1]),
        "std_velocity_z": np.std(concat_velocities[:, 2]),
        "mean_accel_x": np.mean(concat_accelerations[:, 0]),
        "mean_accel_y": np.mean(concat_accelerations[:, 1]),
        "mean_accel_z": np.mean(concat_accelerations[:, 2]),
        "std_accel_x": np.std(concat_accelerations[:, 0]),
        "std_accel_y": np.std(concat_accelerations[:, 1]),
        "std_accel_z": np.std(concat_accelerations[:, 2])
    }

# Print statistics
for key, value in statistics.items():
    print(f"{key}: {value:.7E}")

# Save npz
np.savez_compressed(f"{save_name}.npz", **trajectories)
print(f"npz saved at: ./{save_name}.npz")

# Save metadata.json
if dim == 2:
    metadata = {
        "bounds": bounds,
        "sequence_length": sequence_length,
        "default_connectivity_radius": default_connectivity_radius,
        "boundary_augment": 1.0,
        "material_feature_len": material_feature_len,
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
if dim == 3:
    metadata = {
        "bounds": bounds,
        "sequence_length": sequence_length,
        "default_connectivity_radius": default_connectivity_radius,
        "boundary_augment": 1.0,
        "material_feature_len": material_feature_len,
        "dim": dim,
        "dt": dt_mpm,
        "vel_mean": [statistics["mean_velocity_x"], statistics["mean_velocity_y"], statistics["mean_velocity_z"]],
        "vel_std": [statistics["std_velocity_x"], statistics["std_velocity_y"], statistics["std_velocity_z"]],
        "acc_mean": [statistics["mean_accel_x"], statistics["mean_accel_y"], statistics["mean_accel_z"]],
        "acc_std": [statistics["std_accel_x"], statistics["std_accel_y"], statistics["std_accel_z"]],
        "mpm_cell_size": mpm_cell_size,
        "nparticles_per_cell": nparticles_per_cell,
        "data_names": data_names
    }

with open(f"metadata-{save_name}.json", "w") as fp:
    json.dump(metadata, fp)
print(f"metadata saved at: ./{save_name}.json")


# # See npz
# data = np.load('train.npz', allow_pickle=True)
# for simulation_id, trajectory in data.items():
#     print(simulation_id)
#     print(trajectory)
