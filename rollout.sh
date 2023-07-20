# Note
# Start at the homw working dir--`./gns-mpm`
set -e

# workdir
export DATASET_NAME="sand3d_collision_r030"
#for ((STEPS=10000000; STEPS<14000000; STEPS+=1000000))
for STEPS in 3000000
do
#export STEPS=15270000
for i in {0..4..1}
do
export OUTPUT_TAG="${i}"
export TRAJECTORY_ID="0"

## Change test.npz that is used to make rollout to the specified `test_{OUTPUT_TAG}.npz which you want to make rollout
export WORK_DIR="../gns-mpm-data/gns-data"
if test -f "${WORK_DIR}/datasets/${DATASET_NAME}/test.npz"
 then rm ${WORK_DIR}/datasets/${DATASET_NAME}/test.npz
fi
cp ${WORK_DIR}/datasets/${DATASET_NAME}/trajectory${OUTPUT_TAG}.npz ${WORK_DIR}/datasets/${DATASET_NAME}/test.npz

cd gns-parallel
# Generate test rollouts.
export WORK_DIR="../../gns-mpm-data/gns-data"
python3 -m gns.train \
--mode='rollout' \
--data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" \
--model_path="${WORK_DIR}/models/${DATASET_NAME}/" \
--model_file="model-${STEPS}.pt" \
--train_state_file="train_state-${STEPS}.pt" \
--output_path="${WORK_DIR}/rollouts/${DATASET_NAME}" \
--rollout_filename="rollout_${OUTPUT_TAG}_${TRAJECTORY_ID}_step${STEPS}"
cd ..
# Render rollout
export WORK_DIR="../gns-mpm-data/gns-data"
python3 utils/render_rollout.py \
--output_mode="gif" \
--step_stride=5 \
--rollout_dir="${WORK_DIR}/rollouts/${DATASET_NAME}/" \
--rollout_name="rollout_${OUTPUT_TAG}_${TRAJECTORY_ID}_step${STEPS}"

## Make plots for normalized runout and energy evolution
#export WORK_DIR="../gns-mpm-data/gns-data"
#python3 utils/rollout_analysis.py \
#--rollout_path ${WORK_DIR}/rollouts/${DATASET_NAME} \
#--rollout_filename  rollout_${OUTPUT_TAG}_${TRAJECTORY_ID}_step${STEPS} \
#--output_percentile 100 \
#--output_filename="step${STEPS}_${OUTPUT_TAG}_${TRAJECTORY_ID}"
done
done
