import pickle
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt

# inputs
datasets = ["sand-small-r300-400step_parallel", "sand-small-r300-400step_serial"]


# Plot loss history

locations = []
valid_locations = []
fig, ax = plt.subplots()
color_list = ["black", "skyblue"]
for i, dataset in enumerate(datasets):

    location = f"/work2/08264/baagee/frontera/gns-mpm/gns-data/models/{dataset}/loss_hist.pkl"
    locations.append(location)
    # valid_location = f"/work2/08264/baagee/frontera/gns-mpm/gns-data/models/{dataset}/loss_onestep_train-small-400step-val.pkl"
    # valid_locations.append(valid_location)

    # load train loss and convert it to nparray
    with open(location, 'rb') as f:
        data = pickle.load(f)
    data_nps = []
    for j in range(len(data)):
        data_np = data[j][0], data[j][1].detach().to('cpu').numpy()  # shape=(nsave_steps, 2=(step, loss))
        data_nps.append(data_np)
        # print(data_np)
    # # load validation loss
    # with open(valid_location, 'rb') as f:
    #     validation_loss = pickle.load(f)

    loss_hist = np.array(data_nps)
    print(np.shape(loss_hist))
    #sys.exit()


    ax.plot(loss_hist[:, 0], loss_hist[:, 1], lw=1, alpha=0.5, label=datasets[i])
    # ax.plot(validation_loss[:, 0], validation_loss[:, 1], alpha=0.5, label="validation", color="red")
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_xlim([0, 150000])
    # ax.set_ylim([10e-5, 10])
    ax.legend()
    fig.show()
    fig.savefig(f"/work2/08264/baagee/frontera/gns-mpm/utils/loss{datasets[i]}.png")
