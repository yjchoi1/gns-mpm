import numpy as np

# make arbitrary npz for memory capacity test
trajectories = {}
# nparticles = 18000
# name = f"m{nparticles}"
output = f"/work2/08264/baagee/frontera/gns-mpm-data/gns-data/datasets/sand-small-r300-400step_serial/"

npart_perdim_percell = 4
cellsize = 0.025
offset = cellsize / npart_perdim_percell / 2
xmin = 0.0
xmax = 0.5
ymin = 0.0
ymax = 0.55
particle_interval = cellsize / npart_perdim_percell

# positions = np.random.rand(400, nparticles, 2)
pxs = np.arange(xmin, xmax + offset, particle_interval)
pys = np.arange(ymin, ymax + offset, particle_interval)
p_coords = [[px, py] for px in pxs for py in pys]
p_coords = np.array(p_coords)
nparticles = len(p_coords)
positions = np.stack([p_coords] * 400, axis=0)

print(f"N particles: {nparticles}")

trajectories["trajectory"] = (
    positions,  # position sequence (timesteps, particles, dims)
    np.full(positions.shape[1], 6, dtype=int))

np.savez_compressed(f"{output}/mpm-small-{nparticles}.npz", **trajectories)
print(f"output saved to {output}")
