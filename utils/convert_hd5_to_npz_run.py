from convert_hd5_to_npz import convert_hd5_to_npz

# run
material_feature = True
ndim = 2
dt = 1.0
sim_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions_extrapolation/"
sim_names = [f"extrapol{i}" for i in [0, 1]]
uuid = "/results/extrapol/"

# material_feature = FLAGS.material_feature
# ndim = FLAGS.ndim
# dt = FLAGS.dt
# sim_dir = FLAGS.sim_dir
# sim_name = FLAGS.sim_name
# uuid = FLAGS.sim_name.uuid

for i, sim in enumerate(sim_names):
    convert_hd5_to_npz(path=sim_dir + sim,
                       uuid=uuid,
                       ndim=ndim,
                       output=f"{sim_dir}{sim}/{sim}.npz",
                       material_feature=material_feature,
                       dt=dt
                       )
print("Completed")