set -e

for i in 0
do
python3 animation_from_h5.py \
--path "/work2/08264/baagee/frontera/gns-mpm-data/mpm/mpm-small-test4-3/results/small-test/" \
--output "/work2/08264/baagee/frontera/gns-mpm-data/mpm/mpm-small-test4-3/results" \
--ndim 2 --xboundary 0 0.75 --yboundary 0 0.75
done
#python3 animation_from_h5.py \
#--path "../mpm/3dsand_train${i}/3dsand_train${i}/results/3dsand_train${i}/" \
#--output "../mpm/3dsand_train${i}/3dsand_train${i}/" \
#--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
#done