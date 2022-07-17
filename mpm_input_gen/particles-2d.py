import numpy as np
import math

def create_particle_array(nparticle_per_dir, x_bound, y_bound, dx, dy, ndim=2, randomness=0.8):
    # Geometry
    offset = dx / nparticle_per_dir / 2
    particle_interval = dx / nparticle_per_dir
    xmin = x_bound[0] + offset
    xmax = x_bound[1] - offset
    ymin = y_bound[0] + offset
    ymax = y_bound[1] - offset

    # Create particle arrays
    xs = np.arange(xmin, xmax + offset, particle_interval)
    ys = np.arange(ymin, ymax + offset, particle_interval)
    xy = []
    for y in ys:
        for x in xs:
            xy.append([x, y])
    xy = np.round(xy, 5)
    xy = xy + np.random.uniform(-offset*randomness, offset*randomness, size=xy.shape)
    return xy

#%%

particle_set1 = create_particle_array(
    nparticle_per_dir=4,
    x_bound=[0.40, 0.55], y_bound=[0.1, 0.25],
    dx=0.010, dy=0.010,
    randomness=0.9
)
particles = particle_set1
nparticles = particles.shape[0]

# Write the number of particles
f = open("particles.txt", "w")
f.write(f"{nparticles} \n")
f.close()

# Write coordinates for particles
f = open("particles.txt", "a")
f.write(
    np.array2string(
        particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
    ).replace(' [', '').replace('[', '').replace(']', '')
)
f.close()
