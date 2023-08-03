import numpy as np
import matplotlib.pyplot as plt


def compute_trajectory(particles, velocities, dt, total_time):
    g = 9.81  # acceleration due to gravity in m/s^2
    num_steps = int(total_time / dt)

    # Arrays to store trajectories
    trajectories = np.zeros((particles.shape[0], num_steps, 2))

    for step in range(num_steps):
        t = step * dt

        # Compute x and y using kinematic equations
        dx = velocities[:, 0] * t
        dy = velocities[:, 1] * t - 0.5 * g * t ** 2

        trajectories[:, step, 0] = particles[:, 0] + dx
        trajectories[:, step, 1] = particles[:, 1] + dy

    return trajectories