import torch
import os

# inputs
dir_models = "../gns-data/models/"
dataset = "sand-2d-small2-r300"
model = "train_state-200.pt"

model_path = os.path.join(dir_models, dataset, model)
checkpoint = torch.load(model_path)
print(checkpoint["global_strain_state"]["state"].keys())


