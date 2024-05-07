import pickle
import re
import torch
from matplotlib import pyplot as plt
import numpy as np
import glob

data_path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions3/tall_phi42/outputs_ad_lr500/"
def get_file_number(filename):
    match = re.search(r'(\d+).pt$', filename)
    if match is not None:
        return int(match.group(1))
    return None  # or some suitable default

sorted_files = sorted(glob.glob(f"{data_path}/optimizer_state*.pt"), key=get_file_number)
a= 1
#
epochs = []
time_spent = []
lr_history = []
loss_history = []
friction_history = []
for i, filename in enumerate(sorted_files):
    checkpoint = torch.load(filename)
    epochs.append(checkpoint['epoch'])
    time_spent.append(checkpoint["time_spent"])
    lr_history.append(checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])
    friction_history.append(checkpoint['friction_state_dict']['current_params'].detach().cpu().numpy())
    loss_history.append(checkpoint['loss'].item())

fig, axs = plt.subplots(1, 2)
axs[0].plot(epochs, loss_history)
axs[0].set_title("loss")
axs[1].plot(epochs, friction_history)
axs[1].set_title("friction")
# ax.set_ylim([15, 45])
plt.show()



