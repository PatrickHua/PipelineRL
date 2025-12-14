
export HF_HOME=/ccn2/u/tyhua/.cache/huggingface
export HF_HUB_CACHE=/ccn2/u/tyhua/.cache/huggingface/hub
export RAY_TMPDIR=/ccn2/u/tyhua/ray_tmp
export TMPDIR=/ccn2/u/tyhua/tmp/
export WANDB_DIR=/ccn2/u/tyhua/.cache/wandb
export CUDA_VISIBLE_DEVICES=4,5 


/ccn2/u/tyhua/anaconda3/envs/pipeline-rl/bin/python -m pipelinerl.launch --config-name=multiply output_dir=../pipeline_rl_results/multiply