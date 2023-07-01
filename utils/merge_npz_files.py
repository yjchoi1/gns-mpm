import numpy as np
import json
from tqdm import tqdm

# Inputs
bounds = [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
sequence_length = int(350)
default_connectivity_radius = 0.025
dim = int(3)
material_feature_len = int(0)
dt_mpm = 0.0025  # 0.0025
mpm_cell_size = [None, None, None]  # [0.0125, 0.0125]
nparticles_per_cell = None  # int(16)
dt_gns = 1.0  # 1.0 is default

mpm_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d_collision/"  # "./mpm"
data_case = "trajectory"  # "mpm-9k-train"
data_tags = [i for i in range(0, 3)]
excluded_data_tags = []
data_tags = [i for i in data_tags if i not in excluded_data_tags]
save_name = "sand3d_collisions_train"


trajectories = {}
if dim == 2:
    running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
if dim == 3:
    running_sum = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
    running_sumsq = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
    running_count = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0)
data_names = []

# for computing statistics
cumulative_count = 0
cumulative_sum_vel = np.zeros((1, dim))
cumulative_sum_acc = np.zeros((1, dim))
cumulative_sumsq_vel = np.zeros((1, dim))
cumulative_sumsq_acc = np.zeros((1, dim))

# iterate over each simulation
for id in tqdm(data_tags, total=len(data_tags)):
    data_name = f"{data_case}{id}"
    data_names.append(data_name)
    npz_path = f"{mpm_dir}/{data_name}.npz"  # f"{mpm_dir}/{data_name}/{data_name}.npz"
    data = np.load(npz_path, allow_pickle=True)
    # get trajectory info
    for simulation_id, trajectory in data.items():  # note, only one trajectory exists, so no need to iterate
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

    # Compute statistics
    cumulative_count += len(positions)
    # running sum
    cumulative_sum_vel += np.sum(flattened_velocities, axis=0)
    cumulative_sum_acc += np.sum(flattened_accelerations, axis=0)
    # running sum squared
    cumulative_sumsq_vel += np.sum(flattened_velocities**2, axis=0)
    cumulative_sumsq_acc += np.sum(flattened_accelerations**2, axis=0)
    # statistics for cumulative data
    cumulative_mean_vel = cumulative_sum_vel / cumulative_count
    cumulative_mean_acc = cumulative_sum_acc / cumulative_count
    cumulative_std_vel = np.sqrt(
        (cumulative_sumsq_vel - cumulative_sum_vel ** 2 / cumulative_count) / (cumulative_count - 1))
    cumulative_std_acc = np.sqrt(
        (cumulative_sumsq_acc - cumulative_sum_acc ** 2 / cumulative_count) / (cumulative_count - 1))

# Store final statistics
if dim == 2:
    statistics = {
        "mean_velocity_x": float(cumulative_mean_vel[:, 0]),
        "mean_velocity_y": float(cumulative_mean_vel[:, 1]),
        "std_velocity_x": float(cumulative_std_vel[:, 0]),
        "std_velocity_y": float(cumulative_std_vel[:, 1]),
        "mean_accel_x": float(cumulative_mean_acc[:, 0]),
        "mean_accel_y": float(cumulative_mean_acc[:, 1]),
        "std_accel_x": float(cumulative_std_acc[:, 0]),
        "std_accel_y": float(cumulative_std_acc[:, 1])
    }
if dim == 3:
    statistics = {
        "mean_velocity_x": float(cumulative_mean_vel[:, 0]),
        "mean_velocity_y": float(cumulative_mean_vel[:, 1]),
        "mean_velocity_z": float(cumulative_mean_vel[:, 2]),
        "std_velocity_x": float(cumulative_std_vel[:, 0]),
        "std_velocity_y": float(cumulative_std_vel[:, 1]),
        "std_velocity_z": float(cumulative_std_vel[:, 2]),
        "mean_accel_x": float(cumulative_mean_acc[:, 0]),
        "mean_accel_y": float(cumulative_mean_acc[:, 1]),
        "mean_accel_z": float(cumulative_mean_acc[:, 2]),
        "std_accel_x": float(cumulative_std_acc[:, 0]),
        "std_accel_y": float(cumulative_std_acc[:, 1]),
        "std_accel_z": float(cumulative_std_acc[:, 2])
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
