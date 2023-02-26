import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import os
import sys
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.metrics import mean_squared_error

rollout_tags = [("test2-1", "a=0.5"),
                ("test4", "a=0.8"),
                ("test0-2", "a=1.0"),
                ("test5-1", "a=2.0"),
                ("test5-4", "a=3.0"),
                ("test6-1", "a=4.0"),
                ("test4-3", "Dwonscaled a=0.8"),
                ("test4-2", "Upscaled a=0.8")]
data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand-small-r300-400step_serial/"
training_steps = 15270000
trajectory_ID = [0]
mpm_dt = 0.0025
gravity = 9.81

rollout_filenames = []
for rollout_tag, _ in rollout_tags:
    for i in trajectory_ID:
        if training_steps is not None:
            rollout_filename = f"rollout_{rollout_tag}_{i}_step{training_steps}"
        else:
            rollout_filename = f"rollout_{rollout_tag}_{i}"
        rollout_filenames.append(rollout_filename)


# read positions for MPM and GNS from rollout.pkl
def get_positions(data_path, rollout_filenames):
    trajectories = {}

    for rollout_filename in rollout_filenames:
        with open(os.path.join(data_path, rollout_filename + ".pkl"), "rb") as file:
            # container
            trajectories[rollout_filename] = {
                "mpm": {},
                "gns": {}
            }
            # load rollout
            rollout = pickle.load(file)
            # get positions info
            gns_positions = np.concatenate([rollout["initial_positions"], rollout["predicted_rollout"]], axis=0)
            mpm_positions = np.concatenate([rollout["initial_positions"], rollout["ground_truth_rollout"]], axis=0)
            trajectories[rollout_filename]["mpm"]["positions"] = mpm_positions
            trajectories[rollout_filename]["gns"]["positions"] = gns_positions
            # times
            initial_height = mpm_positions[0, :, 1].max()  # both mpm and gns has to same initial height
            critical_time = np.sqrt(initial_height / 9.81)
            mpm_times = np.linspace(0, len(mpm_positions)*mpm_dt, len(mpm_positions))
            normalized_times = mpm_times / critical_time
            trajectories[rollout_filename]["mpm"]["normalized_times"] = normalized_times
            trajectories[rollout_filename]["gns"]["normalized_times"] = normalized_times

    return trajectories


# Get kinematics info (disp, vel, accel)
def compute_kinemacis(trajectories):
    kinematics_info = {}
    # compute disp, vel, accel from rollout
    for rollout_name, data in trajectories.items():
        kinematics_info[rollout_name] = {}
        for sim, kinematics in data.items():

            # disp
            disps = []
            initial_position = kinematics["positions"][0, :]
            for current_position in kinematics["positions"]:
                displacement = initial_position - current_position
                disp_mag = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)
                disps.append(disp_mag)

            # vels
            velocity = kinematics["positions"][1:, ] - kinematics["positions"][:-1, ]  # velocity for x and y
            # copy initial velocity
            initial_velocity = velocity[0]
            # add third dimension for concat later with velocities
            initial_velocity = np.expand_dims(initial_velocity, 0)
            # concat the velocity to initial vel
            velocity = np.concatenate((velocity, initial_velocity))
            # compute vel magnitude
            velocity_magnitude = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)

            # accels
            accel = velocity[1:, ] - velocity[:-1, ]  # velocity for x and y
            # copy initial velocity
            initial_accel = accel[0]
            # add third dimension for concat later with velocities
            initial_accel = np.expand_dims(initial_accel, 0)
            # concat the velocity to initial vel
            accel = np.concatenate((accel, initial_accel))
            # compute vel magnitude
            accel_magnitude = np.sqrt(accel[:, :, 0] ** 2 + accel[:, :, 1] ** 2)

            # save kinematics in dict
            kinematics_info[rollout_name][sim] = {}
            kinematics_info[rollout_name][sim]["Displacement"] = np.array(disps)
            kinematics_info[rollout_name][sim]["Velocity"] = np.array(velocity_magnitude)
            kinematics_info[rollout_name][sim]["Acceleration"] = np.array(accel_magnitude)

    return kinematics_info


# Compute MSE
def kinematic_error(kinematics_info):
    erros = {}
    for rollout_name, data in kinematics_info.items():
        erros[rollout_name] = {}

        timesteps = len(data["mpm"]["Displacement"])
        for kinematic_type in data["mpm"].keys():
            mse = [
                mean_squared_error(
                    data["gns"][kinematic_type][t], data["mpm"][kinematic_type][t]) for t in range(timesteps)
            ]
            erros[rollout_name][kinematic_type] = mse

    return erros


# get error
trajectories = get_positions(data_path, rollout_filenames)
kinematics_info = compute_kinemacis(trajectories)
error_data = kinematic_error(kinematics_info)


# plot errors for each rollout
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for i, (rollout_name, data) in enumerate(error_data.items()):
    for j, (kinematic_type, value) in enumerate(data.items()):
        normalized_times = trajectories[rollout_name]["mpm"]["normalized_times"]
        ax[j].plot(normalized_times, value, label=f"{rollout_tags[i][1]}")
        ax[j].set_xlabel("timesteps")
        ax[j].set_ylabel(kinematic_type)
    # fig.show()
    plt.tight_layout()
    plt.legend(loc='upper right')
    fig.savefig("error.png")

a = 1




