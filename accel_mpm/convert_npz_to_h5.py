import pandas as pd
import numpy as np
import subprocess


def make_resume_h5_from_npz(
        sample_h5_path,
        to_pass_rollout,
        save_cvs_path,
        out_step=-1,
        dt=0.0025
):

    # get arbitrary sample mpm h5 file to overwrite GNS rollout
    sample_h5_data = pd.read_hdf(sample_h5_path, "/")

    # get position and vel data to pass to h5 for resuming
    gns_positions = np.concatenate((to_pass_rollout["initial_positions"], to_pass_rollout["predicted_rollout"]))
    out_position = gns_positions[out_step]
    out_vel = (gns_positions[out_step] - gns_positions[out_step - 1]) / dt

    # pass GNS prediction to MPM h5 file
    sample_h5_data["coord_x"] = out_position[:, 0]
    sample_h5_data["coord_y"] = out_position[:, 1]
    sample_h5_data["velocity_x"] = out_vel[:, 0]
    sample_h5_data["velocity_y"] = out_vel[:, 1]
    sample_h5_data["stress_xx"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["stress_yy"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["stress_zz"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["tau_xy"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["tau_yz"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["tau_xz"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["strain_xx"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["strain_yy"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["strain_zz"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["gamma_xy"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["gamma_yz"] = np.zeros(out_position[:, 0].shape)
    sample_h5_data["gamma_xz"] = np.zeros(out_position[:, 0].shape)

    # save it as csv, and convert it to h5
    sample_h5_data.to_csv(save_cvs_path)
    shell_command = f"module load intel hdf5;" \
                    f"/work2/08264/baagee/frontera/mpm-csv-hdf5/build/csv-hdf5 {save_cvs_path}"
    try:
        subprocess.run(shell_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the shell script: {e}")
