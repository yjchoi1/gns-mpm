import pickle
import numpy as np
import torch
import sys
sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-material/')
from gns import data_loader

#%%
path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi42/rollout_28_0_step7020000.pkl"
with open(path, "rb") as file:
    rollout_data = pickle.load(file)

initial_pos = rollout_data["initial_positions"]
pred_runout = rollout_data["predicted_rollout"][-1, :, 0].max()
mpm_runout = rollout_data["ground_truth_rollout"][-1, :, 0].max()

error = (pred_runout - mpm_runout)**2

diff = initial_pos[5, :, :] - initial_pos[0, :, :]

#%%
path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi42/sand2d_inverse_eval28.npz"
dinit = data_loader.TrajectoriesDataset(path=path)

for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need `for` loop
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    initial_positions = features[0][:, :6, :]
    particle_type = features[1]
    n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32)

#%%
initial_positions_from_torch = initial_positions.numpy()
initial_positions_from_torch = np.transpose(initial_positions_from_torch, (1, 0, 2))
diff_pos = initial_positions_from_torch - initial_pos

# There is no diff between initial pos when rollout vs original npz
