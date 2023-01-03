import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt

# file_loc = "gns-data/rollouts/sand-small-r300-400step_parallel"
# rollout_case = "rollout_test5-1_0"
# steps = range(0, 10000000, 1000000)
rollout_case = "rollout_test4-2_0"
steps = [500000, 2000000, 4000000]

def append_loss(file_loc, rollout_case, steps):
    loss_with_steps = []
    for step in steps:
        with open(f"{file_loc}/{rollout_case}_step{step}.pkl", "rb") as file:
            rollout_data = pickle.load(file)
        loss = step, rollout_data["loss"].cpu().numpy()
        loss_with_steps.append(loss)
    return loss_with_steps

rollout_loss_26 = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300-400step_parallel',
    rollout_case=rollout_case,
    steps=steps
)

rollout_loss_18 = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300_parallel_dwnsmp18',
    rollout_case=rollout_case,
    steps=steps
)

rollout_loss_11 = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300_parallel_dwnsmp11',
    rollout_case=rollout_case,
    steps=steps
)

rollout_loss_7 = append_loss(
    file_loc='gns-data/rollouts/sand-small-r300_parallel_dwnsmp7',
    rollout_case=rollout_case,
    steps=steps
)

loss_by_num_trj_list = [
    np.array(rollout_loss_26),
    np.array(rollout_loss_18),
    np.array(rollout_loss_11),
    np.array(rollout_loss_7)
]
loss_by_num_trj = np.array(loss_by_num_trj_list)
# loss_by_num_trj = np.transpose(loss_by_num_trj, (1, 0, 2))

fig, ax = plt.subplots()
markers = ['>', 's', 'o']
ax.plot([26, 18, 11, 7], loss_by_num_trj[:, 0, 1], label="0.5M", marker=">")
ax.plot([26, 18, 11, 7], loss_by_num_trj[:, 1, 1], label="2M", marker="s")
ax.plot([26, 18, 11, 7], loss_by_num_trj[:, 2, 1], label="4M", marker="o")
ax.set_xlabel("Number of trajectories")
ax.set_ylabel("Rollout loss")
ax.set_title(f"{rollout_case}")
ax.legend()
plt.show()
fig.savefig(f"/work2/08264/baagee/frontera/gns-mpm/utils/rollout_loss_{rollout_case}.png")





