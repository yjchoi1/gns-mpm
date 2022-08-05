# # venv
source ~/python_envs/venv-gns-cpu/bin/activate

# # Note
#/scratch/04709/vantaj94/gns_pytorch/Sand
# /work/08264/baagee/ls6/gns-data-2d-column

cd gns

# workdir
export WORK_DIR="../gns-data"
export DATASET_NAME="sand-2d"

# Generate test rollouts.
python3 -m gns.train --mode='rollout' --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --model_file="model-deepmind-10000000.pt" --train_state_file="train_state-deepmind-10000000.pt" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}"

# Render rollout
python3 -m gns.render_rollout --rollout_path="${WORK_DIR}/rollouts/${DATASET_NAME}/rollout_0.pkl"

cd ..
