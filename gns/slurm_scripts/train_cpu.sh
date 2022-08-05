# start in slurm_scripts
source ~/python_envs/venv-gns-cpu/bin/activate
	
# assume data is already downloaded and hardcode WaterDropSample

export DATASET_NAME="sand-2d"
export WORK_DIR="../gns-data"

python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" -ntraining_steps=100

