
export HF_HOME=/ccn2/u/tyhua/.cache/huggingface
export HF_HUB_CACHE=/ccn2/u/tyhua/.cache/huggingface/hub
export RAY_TMPDIR=/ccn2/u/tyhua/ray_tmp
export TMPDIR=/ccn2/u/tyhua/tmp/
export WANDB_DIR=/ccn2/u/tyhua/.cache/wandb
export CUDA_VISIBLE_DEVICES=0,1 

conda activate pipeline-rl

python -m pipelinerl.launch --config-name=guessing output_dir=results/guessing