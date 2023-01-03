import h5py

dir = "/work2/08264/baagee/frontera/gns-mpm/mpm/3d-sand8/3d-sand8/results/3d-sand8/"
fnames = ["particles100000.h5"]
# get size of trajectory
with h5py.File(dir+fnames[0], "r") as f:
    (nparticles,) = f["table"]["coord_x"].shape
    # print(nparticles)
    # print(f["table"].shape)
    x = f["table"]["coord_x"][:]
nsteps = len(fnames)