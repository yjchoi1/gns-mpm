set -e

for i in 6
do
python3 animation_from_h5.py \
--path "../mpm/3d-sand${i}/3d-sand${i}/results/3d-sand${i}/" \
--output "../mpm/3d-sand${i}/3d-sand${i}/results/" \
--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
done
