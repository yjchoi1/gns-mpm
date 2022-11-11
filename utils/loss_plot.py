import pickle
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt

# inputs
datasets = ["sand-small-r300-400step_serial", "sand-small-r300-400step_parallel"]

fig, ax = plt.subplots()
# Plot loss history
locations = []
for i, dataset in enumerate(datasets):
    location = f"/work2/08264/baagee/frontera/gns-mpm/gns-data/models/{dataset}/loss_hist.pkl"
    locations.append(location)
    with open(location, 'rb') as f:
        data = pickle.load(f)
    data_nps = []
    for j in range(len(data)):
        data_np = data[j][0], data[j][1].detach().to('cpu').numpy()  # shape=(nsave_steps, 2=(step, loss))
        data_nps.append(data_np)
        # print(data_np)

    loss_hist = np.array(data_nps)
    print(np.shape(loss_hist))
    #sys.exit()
    
    ax.plot(loss_hist[:, 0], loss_hist[:, 1], lw=1, alpha=0.5, label=datasets[i])
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.show()
    fig.savefig("loss.png")
