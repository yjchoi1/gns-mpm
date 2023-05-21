import argparse
import pathlib
import glob
import re
import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def from_h5_to_animation():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", help="Location of folder containing mpm vtk files")
    parser.add_argument("--output", default="mpm/mpm-6k-train0/results", help="Location to save the animation")
    parser.add_argument("--ndim", default=2, help="dimension of the simulation")
    parser.add_argument("--xboundary", nargs="+", help="x boundary of simulation domain")
    parser.add_argument("--yboundary", nargs="+", help="y boundary of simulation domain")
    parser.add_argument("--zboundary", nargs="+", default=None, help="z boundary of simulation domain")
    args = parser.parse_args()

    path = args.path
    output = args.output
    xboundary = args.xboundary
    yboundary = args.yboundary
    ndim = args.ndim

    # check file existence at "path"
    directories = [pathlib.Path(path) for path in path]
    for directory in directories:
        if not directory.exists():
            raise FileExistsError(f"The path {directory} does not exist.")
    print(f"Number of trajectories: {len(directories)}")

    # dimension of the MPM simulation
    ndim = int(ndim)

    trajectories = {}
    for nth_trajectory, directory in enumerate(directories):
        # read `.h5`
        fnames = glob.glob(f"{str(directory)}/*.h5")
        get_fnumber = re.compile(".*\D(\d+).h5")
        fnumber_and_fname = [(int(get_fnumber.findall(fname)[0]), fname) for fname in fnames]
        fnumber_and_fname_sorted = sorted(fnumber_and_fname, key=lambda row: row[0])

        # get size of trajectory
        with h5py.File(fnumber_and_fname_sorted[0][1], "r") as f:
            (nparticles,) = f["table"]["coord_x"].shape
            # print(nparticles)
            # print(f["table"].shape)
        nsteps = len(fnames)
        # allocate memory for trajectory
        # assume number of particles does not change along the rollout.
        positions = np.empty((nsteps, nparticles, ndim), dtype=float)
        print(f"Size of trajectory {nth_trajectory} ({directory}): {positions.shape}")
        # print(np.shape(positions))

        # open each file and copy data to positions tensor.
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                for idx, name in zip(range(ndim), ["coord_x", "coord_y", "coord_z"]):
                    missing_particles = nparticles - len(f["table"][name][:])
                    # get position of each step but pad zero if particles are missing
                    positions[nth_step, :, idx] = np.pad(f["table"][name][:], (0, missing_particles), 'constant')

    if ndim == 2:
        # make animation
        fig, ax = plt.subplots()

        def animate(i):
            fig.clear()
            xboundary = args.xboundary
            yboundary = args.yboundary
            # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
            ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
            ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], s=1)
            ax.grid(True, which='both')

        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, len(positions), 3), interval=10)

        ani.save(f'{output}/trajectory.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {output}")

    if ndim == 3:
        # make animation
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def animate(i):
            fig.clear()
            xboundary = args.xboundary
            yboundary = args.yboundary
            zboundary = args.zboundary
            # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
            ax = fig.add_subplot(projection='3d', autoscale_on=False)
            ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
            ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
            ax.set_zlim([float(zboundary[0]), float(zboundary[1])])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2], s=1)
            ax.set_box_aspect(
                aspect=(float(xboundary[1]) - float(xboundary[0]),
                        float(yboundary[1]) - float(yboundary[0]),
                        float(zboundary[1]) - float(zboundary[0])))
            ax.view_init(elev=20., azim=i*0.5)
            ax.grid(True, which='both')

        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, len(positions), 3), interval=10)

        ani.save(f'{output}/trajectory.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {output}")

if __name__ == "__main__":
    from_h5_to_animation()