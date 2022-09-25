import pickle
import numpy as np
import torch

# Compare loss history bewtween the GNS with gravity constriant and without it
location_no_g = "/work2/08264/baagee/frontera/gns-mpm/gns-data/models/sand-2d-small-gravity-r300/loss_hist.pkl"

with open(location_no_g, 'rb') as f:
    data = pickle.load(f)

#data2 = data.detach().to('cpu').numpy()
#print(data2[0])
#print(data[0][1].detach().to('cpu').numpy())


data_nps = []
for i in range(len(data)):
	data_np = data[i][0], data[i][1].detach().to('cpu').numpy()
	data_nps.append(data_np)

print(data_nps[0])

with open('loss_hist_with_g.pkl', 'wb') as f:
	pickle.dump(data_nps, f)
    

