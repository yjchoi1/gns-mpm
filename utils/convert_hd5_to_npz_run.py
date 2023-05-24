from convert_hd5_to_npz import convert_hd5_to_npz

# run
material_feature = False
ndim = 3
dt = 1.0
sim_dir = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d/"
sim_names = [f"sand3d_column_collapse{i}" for i in [8, 9, 10, 11, 12, 13, 14, 15, 16]]
uuid = "/results/sand3d_column_collapse/"

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