
export HF_HOME=/ccn2/u/tyhua/.cache/huggingface
export HF_HUB_CACHE=/ccn2/u/tyhua/.cache/huggingface/hub
export RAY_TMPDIR=/ccn2/u/tyhua/ray_tmp
export TMPDIR=/ccn2/u/tyhua/tmp/
export WANDB_DIR=/ccn2/u/tyhua/.cache/wandb
export TRITON_CACHE_DIR=/tmp/triton_cache_$(whoami)  # Use local /tmp instead of NFS
export CUDA_VISIBLE_DEVICES=4,5 

export EXPERIMENT_NAME=multiply_debug

# Create output directory if it doesn't exist
mkdir -p ../pipeline_rl_results/$EXPERIMENT_NAME

/ccn2/u/tyhua/anaconda3/envs/pipeline-rl/bin/python -m pipelinerl.launch \
    --config-name=multiply output_dir=../pipeline_rl_results/$EXPERIMENT_NAME 2>&1 | tee ../pipeline_rl_results/$EXPERIMENT_NAME/log.txt