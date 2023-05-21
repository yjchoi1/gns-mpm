import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# read positions for MPM and GNS from rollout.pkl
def get_positions(data_path, rollout_filenames, mpm_dt):
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
            initial_height = mpm_positions[0, :, -1].max()  # both mpm and gns has to same initial height
            critical_time = np.sqrt(initial_height / 9.81)
            mpm_times = np.linspace(0, len(mpm_positions)*mpm_dt, len(mpm_positions))
            normalized_times = mpm_times / critical_time
            trajectories[rollout_filename]["mpm"]["normalized_times"] = normalized_times
            trajectories[rollout_filename]["gns"]["normalized_times"] = normalized_times

            # simulation metadata
            metadata = rollout["metadata"]

    return trajectories, metadata


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
            ndims = initial_position.shape[1]
            for current_position in kinematics["positions"]:
                displacement = initial_position - current_position
                if ndims == 3:  # if data is 3d
                    disp_mag = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2 + displacement[:, 2] ** 2)
                elif ndims == 2:  # if data is 2d
                    disp_mag = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)
                else:
                    NotImplementedError("Displacement data should be 2d or 3d")
                disps.append(disp_mag)
            disps_magnitude = np.array(disps)

            # vels
            velocity = kinematics["positions"][1:, ] - kinematics["positions"][:-1, ]  # velocity for x and y
            # copy initial velocity
            initial_velocity = velocity[0]
            # add third dimension for concat later with velocities
            initial_velocity = np.expand_dims(initial_velocity, 0)
            # concat the velocity to initial vel
            velocity = np.concatenate((velocity, initial_velocity))
            # compute vel magnitude
            if ndims == 2:
                velocity_magnitude = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)
            elif ndims == 3:
                # print("3d velocity")
                velocity_magnitude = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2, velocity[:, :, 2] ** 2)
                a=1
            else:
                NotImplementedError("Velocity data should be 2d or 3d")

            # accels
            accel = velocity[1:, ] - velocity[:-1, ]  # velocity for x and y
            # copy initial velocity
            initial_accel = accel[0]
            # add third dimension for concat later with velocities
            initial_accel = np.expand_dims(initial_accel, 0)
            # concat the velocity to initial vel
            accel = np.concatenate((accel, initial_accel))
            # compute vel magnitude
            if ndims == 2:
                accel_magnitude = np.sqrt(accel[:, :, 0] ** 2 + accel[:, :, 1] ** 2)
            elif ndims == 3:
                accel_magnitude = np.sqrt(accel[:, :, 0] ** 2 + accel[:, :, 1] ** 2 + accel[:, :, 2] ** 2)

            # save kinematics in dict
            kinematics_info[rollout_name][sim] = {}
            kinematics_info[rollout_name][sim]["displacement"] = np.array(disps_magnitude)
            kinematics_info[rollout_name][sim]["velocity"] = np.array(velocity_magnitude)
            kinematics_info[rollout_name][sim]["acceleration"] = np.array(accel_magnitude)

    return kinematics_info


# Compute MSE
def kinematic_error(kinematics_info):
    erros = {}
    individual_error = {}
    mape = {}

    for rollout_name, data in kinematics_info.items():
        erros[rollout_name] = {}
        individual_error[rollout_name] = {}
        mape[rollout_name] = {}

        timesteps = len(data["mpm"]["displacement"])

        # MSE for entire particles
        for kinematic_type in data["mpm"].keys():
            mse = [
                mean_squared_error(
                    data["gns"][kinematic_type][t], data["mpm"][kinematic_type][t]) for t in range(timesteps)
            ]
            # if kinematic_type == "displacement":
            #     static_portion = sum(data["gns"][kinematic_type][-1] < 0.012) / len(data["gns"][kinematic_type])
            erros[rollout_name][kinematic_type] = np.array(mse)

        # MSE for individual particles
        for kinematic_type in data["mpm"].keys():
            individual_mse = [
                (data["gns"][kinematic_type][t] - data["mpm"][kinematic_type][t])**2/2
                 for t in range(timesteps)
            ]
            individual_error[rollout_name][kinematic_type] = np.array(individual_mse)

        # MAPE for entire particles
        for kinematic_type in data["mpm"].keys():
            mape_temp = [
                # np.mean(np.abs(data["mpm"][kinematic_type][t] - data["gns"][kinematic_type][t]) / data["mpm"][kinematic_type][t]) * 100
                mean_absolute_percentage_error(
                    data["gns"][kinematic_type][t], data["mpm"][kinematic_type][t]) for t in range(timesteps)
            ]
            mape[rollout_name][kinematic_type] = np.array(mape_temp)


    return erros, individual_error, mape
