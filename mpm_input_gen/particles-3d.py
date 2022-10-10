import numpy as np
import math

def create_particle_array(ndim, nparticle_per_dir, x_bound, y_bound, z_bound, dx, dy, dz):
    # Geometry
    offset = dx / nparticle_per_dir / 2
    particle_interval = dx / nparticle_per_dir
    xmin = x_bound[0] + offset
    xmax = x_bound[1] - offset
    ymin = y_bound[0] + offset
    ymax = y_bound[1] - offset
    zmin = z_bound[0] + offset
    zmax = z_bound[1] - offset

    # Create particle arrays
    x = np.arange(xmin, xmax + offset, particle_interval)
    y = np.arange(ymin, ymax + offset, particle_interval)
    z = np.arange(zmin, zmax + offset, particle_interval)
    xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, ndim)
    return xyz
#%%

particle_set1 = create_particle_array(
    ndim=3, nparticle_per_dir=4,
    x_bound=[0.1, 0.4], y_bound=[0.1, 0.4], z_bound=[0.3, 0.6],
    dx=0.01, dy=0.01, dz=0.01)

particle_set2 = create_particle_array(
    ndim=3, nparticle_per_dir=4,
    x_bound=[0.6, 0.9], y_bound=[0.6, 0.9], z_bound=[0.3, 0.6],
    dx=0.01, dy=0.01, dz=0.01)

# particles = np.concatenate((particle_set1, particle_set2))
particles = particle_set1

nparticles = particles.shape[0]

#%%

# Write the number of particles
f = open("particles.txt", "w")
f.write(f"{nparticles} \n")
f.close()

# Write coordinates for particles
f = open("particles.txt", "a")
f.write(np.array2string(particles, threshold=math.inf).replace(' [', '').replace('[', '').replace(']', ''))
f.close()

#%% Delete

# Input
ndim = 3
nparticle_per_dir = 2  # number of particle per direction in one element
x_bound = [0.1, 0.4]  # Boundaries for entitiy that contains particles
y_bound = [0.1, 0.4]
z_bound = [0.6, 0.9]
dx = 0.1  # Element length
dy = 0.1
dz = 0.1

# Geometry
nele_x = abs(x_bound[0] - x_bound[1])/dx
nele_y = abs(y_bound[0] - y_bound[1])/dy
nele_z = abs(z_bound[0] - z_bound[1])/dz
nele = nele_x*nele_y*nele_z
offset = dx/nparticle_per_dir/2
particle_interval = dx/nparticle_per_dir
xmin = x_bound[0] + offset
xmax = x_bound[1] - offset
ymin = y_bound[0] + offset
ymax = y_bound[1] - offset
zmin = z_bound[0] + offset
zmax = z_bound[1] - offset

# Create particle arrays
x = np.arange(xmin, xmax+offset, particle_interval)
y = np.arange(ymin, ymax+offset, particle_interval)
z = np.arange(zmin, zmax+offset, particle_interval)
xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, ndim)
nparticles = xyz.shape[0]

# Write the number of particles
f = open("particles.txt", "w")
f.write(f"{nparticles} \n")
f.close()

# Write coordinates for particles
f = open("particles.txt", "a")
f.write(np.array2string(xyz, threshold=math.inf).replace(' [', '').replace('[', '').replace(']', ''))
f.close()