import numpy as np
import glob

npz_dirs = sorted(glob.glob("/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/lbm-pipe/separate_npzs/*npz"))
# _data = [item for _, item in np.load(npz_dirs[0], allow_pickle=True).items()]
data_merged = {}
for i, npz_dir in enumerate(npz_dirs[: 20]):
    print(i)
    data = dict(np.load(npz_dir, allow_pickle=True))
    data_merged.update(data)
print("Merging")
np.savez_compressed(
    f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/lbm-pipe/mgd.npz", **data_merged)

# #%%
# import numpy as np
# import h5py
# import glob
# path = "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/lbm-pipe/separate_npzs/"
# def merge_npz_to_hdf5(start, end, step, hdf5_filename):
#     # Open or create the HDF5 file
#     with h5py.File(hdf5_filename, 'a') as f:
#         for j in range(start, end, step):
#             npz_filename = f"{path}/{j}_to_{j+step}.npz"
#             print(f"Processing {npz_filename}")
#
#             # Load the .npz file
#             with np.load(npz_filename, allow_pickle=True) as npz_data:
#                 for key, value in npz_data.items():
#                     # Check if dataset already exists in the HDF5 file
#                     if key in f:
#                         # Append data to existing dataset
#                         old_size = f[key].shape[0]
#                         new_size = old_size + value.shape[0]
#                         f[key].resize(new_size, axis=0)
#                         f[key][old_size:] = value
#                     else:
#                         # Create a new dataset
#                         maxshape = (None,) + value.shape[1:]
#                         f.create_dataset(key, data=value, maxshape=maxshape, chunks=True)
#
# merge_npz_to_hdf5(0, 1000, 20, "merged_data.h5")
