from convert_hd5_to_npz import convert_hd5_to_npz

# run
material_feature = True
ndim = 2
dt = 1.0
sim_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/"
sim_names = [f"sand2d_frictions_test{i}" for i in range(0, 7)] + [f"sand2d_frictions_test{i}" for i in range(8, 14)]
uuid = "/results/sand2d_frictions_test"

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