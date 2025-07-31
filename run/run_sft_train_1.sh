# run script for SFT training

# export HF_HOME=~/.cache/huggingface
# export TRANSFORMERS_CACHE=~/.cache/huggingface



# CUDA_VISIBLE_DEVICES=1,3 accelerate launch -m scripts.train_sft
# CUDA_VISIBLE_DEVICES=1,3 python -m scripts.train_sft
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.train_mp_sft --devices 0 1 2 3
