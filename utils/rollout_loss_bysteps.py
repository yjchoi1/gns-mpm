import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt

# file_loc = "gns-data/rollouts/sand-small-r300-400step_parallel"
# rollout_case = "rollout_test5-1_0"
# steps = range(0, 10000000, 1000000)

def append_loss(file_loc, rollout_case, steps):
    loss_with_steps = []
    for step in steps:
        with open(f"{file_loc}/{rollout_case}_step{step}.pkl", "rb") as file:
            rollout_data = pickle.load(file)
        loss = step, rollout_data["loss"].cpu().numpy()
        loss_with_steps.append(loss)
    return loss_with_steps

rollout_loss_parallel = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300-400step_parallel',
    rollout_case='rollout_test4-2_0',
    steps=range(1000000, 14000000, 1000000)
)

rollout_loss_serial = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300-400step_serial',
    rollout_case='rollout_test4-2_0',
    steps=range(1000000, 14000000, 1000000)
)

parallel = np.array(rollout_loss_parallel)
serial = np.array(rollout_loss_serial)
plt.plot(parallel[:, 0], parallel[:, 1], label="parallel")
plt.plot(serial[:, 0], serial[:, 1], label="serial")
plt.xlabel("Training step")
plt.ylabel("Rollout positional loss")
plt.legend()
plt.savefig("rollout_loss.png")

plt.show()




