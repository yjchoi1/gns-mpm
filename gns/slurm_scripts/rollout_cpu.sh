# start in slurm_scripts
source ~/python_envs/venv-gns-cpu/bin/activate
	
# assume data is already downloaded and hardcode WaterDropSample

export DATASET_NAME="sand-2d"
export WORK_DIR="../gns-data"

python3 -m gns.train --mode="rollout" --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --model_file="model-deepmind-10000000.pt" --train_state_file="train_state-deepmind-10000000.pt" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/"

