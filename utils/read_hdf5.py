import h5py

dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/sand2d_frictions83/results/sand2d_frictions/"
fnames = ["particles300000.h5"]
# get size of trajectory
with h5py.File(dir+fnames[0], "r") as f:
    (nparticles,) = f["table"]["coord_x"].shape
    # print(nparticles)
    # print(f["table"].shape)
    x = f["table"]["coord_x"][:]
nsteps = len(fnames)