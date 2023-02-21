set -e

for i in 0
do
python3 animation_from_h5.py \
--path "../../gns-mpm-data/mpm/sand3d/sand3d101/results/sand3d/" \
--output "../../gns-mpm-data/mpm/sand3d/sand3d101" \
--ndim 3 --xboundary -0.0208334 1.0208334 --yboundary -0.0208334 1.0208334 --zboundary -0.0208334 1.0208334
done
#python3 animation_from_h5.py \
#--path "../mpm/3dsand_train${i}/3dsand_train${i}/results/3dsand_train${i}/" \
#--output "../mpm/3dsand_train${i}/3dsand_train${i}/" \
#--ndim 3 --xboundary 0 1 --yboundary 0 1 --zboundary 0 1
#done