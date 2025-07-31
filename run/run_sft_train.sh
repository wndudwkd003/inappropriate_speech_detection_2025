# run script for SFT training

# export HF_HOME=~/.cache/huggingface
# export TRANSFORMERS_CACHE=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_sft --config configs
