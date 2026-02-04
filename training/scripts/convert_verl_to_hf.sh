JOB_WORKING_DIR="/home/ianwu/code/verl-stable/scripts"
MODEL_PATH="/tmp/ianwu/checkpoints/reasoning_cache/online_acemath_rl_4b_inst_hard_16k_self_verify/global_step_100/actor"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" # Verify Slurm is parsing --gres correctly
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"


cd $JOB_WORKING_DIR

mkdir -p $MODEL_PATH/huggingface
cp /tmp/ianwu/huggingface/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/*.json $MODEL_PATH/huggingface/ 

python convert_fsdp_to_hf.py $MODEL_PATH $MODEL_PATH/huggingface $MODEL_PATH/hf-format 8
