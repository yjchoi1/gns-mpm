set -e

for i in 0
do
python3 animation_from_h5.py \
--path "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d/sand3d_column_collapse0/results/sand3d_column_collapse" \
--output "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d/sand3d_column_collapse0/" \
--ndim 3 --xboundary  -0.02083 1.02083 --yboundary -0.1458333 0.1458333 --zboundary -0.02083 1.02083
done
#python3 animation_from_h5.py \
#--path "../mpm/3dsand_train${i}/3dsand_train${i}/results/3dsand_train${i}/" \
#--output "../mpm/3dsand_train${i}/3dsand_train${i}/" \
#--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
#done