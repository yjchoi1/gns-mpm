import numpy as np
import pickle
from matplotlib import pyplot as plt
import os


# %% inputs for read rollout.pkl
rollout_path = "../../gns-data/rollouts/sand-2d-small-r300"
rollout_filename = "rollout_a=1.pkl"
# inputs for result
timesteps = 320
g = 9.81 * 0.0025**2  # because, in MPM, we used dt=0.0025, and, in GNS, dt=1.0
mass = 1


def analyze_runout(rollout_path, rollout_filename, timesteps=320, g=6.13125e-05, mass=1):
    print("a")
    return


# %% Read rollout data
with open(os.path.join(rollout_path, rollout_filename), "rb") as file:
    rollout_data = pickle.load(file)
predicted_trajectory = np.concatenate([
    rollout_data["initial_positions"],
    rollout_data["predicted_rollout"]], axis=0)
mpm_trajectory = np.concatenate([
    rollout_data["initial_positions"],
    rollout_data["ground_truth_rollout"]], axis=0)
trajectories = (mpm_trajectory, predicted_trajectory)

# %% Extract data from runout
for i, trajectory in enumerate(trajectories):

    # velocities
    velocities = trajectory[1:, ] - trajectory[:-1, ]  # velocities for x and y
    scalar_velocities = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2)  # scalar sum of x y velocities
    # runout height (H) and length (L)
    L_t = []
    H_t = []
    for position in trajectory:
        L = np.amax(position[:, 0])  # front end of runout
        H = np.amax(position[:, 1])  # top of runout
        L_t.append(L)
        H_t.append(H)
    # normalize H, L with initial length of column and time with critical time
    L_initial = np.amax(trajectory[0][:, 0]) - np.min(trajectory[0][:, 0])
    H_initial = np.amax(trajectory[0][:, 1]) - np.min(trajectory[0][:, 1])
    critial_time = np.sqrt(H_initial/g)
    time = np.arange(0, timesteps, 1)
    normalized_time = time/critial_time
    normalized_L_t = (L_t - L_initial)/L_initial
    normalized_H_t = H_t/L_initial
    # compute energies
    potentialE = np.sum(mass*g*trajectory[:, :, 1], axis=-1)  # sum(mass * gravity * elevation)
    kineticE = (1/2) * np.sum(mass*scalar_velocities**2, axis=-1)
    E0 = potentialE[0] + kineticE[0]
    dissipationE = E0 - kineticE - potentialE[1:]
    # normalize energies
    normalized_Ek = kineticE/E0
    normalized_Ep = potentialE/E0
    normalized_Ed = dissipationE/E0


    # %% Plot
    if i == 0:
        fig, axs = plt.subplots(3, 2)
    else:
        pass
    # plot
    legends = ["mpm", "gns"]
    axs[0, 0].plot(normalized_time, normalized_L_t, label=legends[i])
    axs[0, 1].plot(normalized_time, normalized_H_t)
    axs[1, 0].plot(normalized_time, normalized_Ep)
    axs[1, 1].plot(normalized_time[1:], normalized_Ek)
    axs[2, 0].plot(normalized_time[1:], normalized_Ed)
    # labels
    for ax in axs.flatten():
        ax.set_xlabel("$t / \sqrt{H/g}$")
    axs[0, 0].set_ylabel("$(L_t - L)/L$")
    axs[0, 1].set_ylabel("$H_t/L$")
    axs[1, 0].set_ylabel("$E_p /E_0$")
    axs[1, 1].set_ylabel("$E_k /E_0$")
    axs[2, 0].set_ylabel("$E_d /E_0$")
    # axs[0, 0].set_ylim(bottom=0, top=5)
    # axs[0, 1].set_ylim(bottom=0, top=1.05)
    axs[1, 0].set_ylim(bottom=0, top=1.05)
    # axs[1, 1].set_ylim(bottom=0, top=0.05)
    # axs[2, 0].set_ylim(bottom=0, top=0.3)

    axs[0,0].legend()
    plt.tight_layout()

    fig.show()
