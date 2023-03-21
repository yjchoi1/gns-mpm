set -e

for i in 68
do
python3 animation_from_h5.py \
--path "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/sand2d_frictions_test1/results/sand2d_frictions_test/" \
--output "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions/sand2d_frictions_test1/" \
--ndim 2 --xboundary -0.0025 1.0025 --yboundary -0.0025 1.0025
done
#python3 animation_from_h5.py \
#--path "../mpm/3dsand_train${i}/3dsand_train${i}/results/3dsand_train${i}/" \
#--output "../mpm/3dsand_train${i}/3dsand_train${i}/" \
#--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
#done