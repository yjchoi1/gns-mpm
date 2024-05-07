set -e

for i in 0
do
python3 animation_from_h5.py \
--path "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions_extrapolation/extrapol0/results/extrapol/" \
--output "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand2d_frictions_extrapolation/extrapol0/" \
--ndim 2 --xboundary 0.0 1.0 --yboundary 0.0 0.5 \
--step_stride 5
done
#python3 animation_from_h5.py \
#--path "../mpm/3dsand_train${i}/3dsand_train${i}/results/3dsand_train${i}/" \
#--output "../mpm/3dsand_train${i}/3dsand_train${i}/" \
#--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
#done