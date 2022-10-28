import numpy as np

data = np.load(f'train.npz', allow_pickle=True)

for i, p in data.items():
    print(i, p)