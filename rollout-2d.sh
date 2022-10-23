# Note
# Start at the homw working dir--`./gns-mpm`
set -e

# workdir
export DATASET_NAME="sand-2d-small-r300"
export STEPS=2000000
export OUTPUT_TAG="test4-1"
export TRAJECTORY_ID="0"

# Change test.npz that is used to make rollout to the specified `test_{OUTPUT_TAG}.npz which you want to make rollout
export WORK_DIR="gns-data"
if test -f "${WORK_DIR}/datasets/${DATASET_NAME}/test.npz"
 then rm gns-data/datasets/${DATASET_NAME}/test.npz
fi
cp ${WORK_DIR}/datasets/${DATASET_NAME}/mpm-small-${OUTPUT_TAG}.npz ${WORK_DIR}/datasets/${DATASET_NAME}/test.npz

cd gns
# Generate test rollouts.
export WORK_DIR="../gns-data"
python3 -m gns.train --mode='rollout' --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --model_file="model-${STEPS}.pt" --train_state_file="train_state-${STEPS}.pt" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}" --rollout_tag=${OUTPUT_TAG}
cd ..

# Render rollout
export WORK_DIR="gns-data"
python3 utils/render_rollout.py --rollout_path="${WORK_DIR}/rollouts/${DATASET_NAME}" --rollout_filename="rollout_${OUTPUT_TAG}_${TRAJECTORY_ID}" --output_filename="rollout_trainstep${STEPS}_${OUTPUT_TAG}_${TRAJECTORY_ID}"

# Make plots for normalized runout and energy evolution
export WORK_DIR="gns-data"
python3 utils/rollout_analysis.py --rollout_path ${WORK_DIR}/rollouts/${DATASET_NAME} --rollout_filename  rollout_${OUTPUT_TAG}_${TRAJECTORY_ID} --output_percentile 100 --output_filename="trainstep${STEPS}_${OUTPUT_TAG}_${TRAJECTORY_ID}"