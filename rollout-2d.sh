# # Note
#/scratch/04709/vantaj94/gns_pytorch/Sand
# /work/08264/baagee/ls6/gns-data-2d-column

cd gns

# workdir
export WORK_DIR="../gns-data"
export DATASET_NAME="sand-2d-small3-r300"
export STEPS=2020000
export OUTPUT_TAG="test0-2"
export TRAJECTORY_ID="0"

# Change test.npz that is used to make rollout to the specified `test_{OUTPUT_TAG}.npz which you want to make rollout
rm ../gns-data/datasets/${DATASET_NAME}/test.npz
cp ../gns-data/datasets/${DATASET_NAME}/mpm-small-${OUTPUT_TAG}.npz ../gns-data/datasets/${DATASET_NAME}/test.npz

# Generate test rollouts.
python3 -m gns.train --mode='rollout' --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --model_file="model-${STEPS}.pt" --train_state_file="train_state-${STEPS}.pt" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}" --rollout_tag=${OUTPUT_TAG}

# Render rollout
python3 -m gns.render_rollout --rollout_path="${WORK_DIR}/rollouts/${DATASET_NAME}/rollout_${OUTPUT_TAG}_${TRAJECTORY_ID}.pkl" --dataset_name=${DATASET_NAME} --output_tag=${OUTPUT_TAG}_${TRAJECTORY_ID}

# Make plots for normalized runout and energy evolution
cd ..
export WORK_DIR="gns-data"
python3 utils/rollout_analysis.py --rollout_path ${WORK_DIR}/rollouts/${DATASET_NAME} --rollout_filename  rollout_${OUTPUT_TAG}_${TRAJECTORY_ID} --output_percentile 100