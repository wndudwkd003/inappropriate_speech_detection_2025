# run script for SFT training

# export HF_HOME=~/.cache/huggingface
# export TRANSFORMERS_CACHE=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=1,2 python -m scripts.train_dpo --path "output/2025-07-28_14-04-14_kakaocorp_kanana-1.5-8b-base_r_128_ra_128_rd_0.0_float16_sft"
