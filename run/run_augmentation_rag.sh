#!/bin/bash


# PYTHONPATH 설정과 함께 실행
CUDA_VISIBLE_DEVICES=0 python -m scripts.rag.augment_with_rag --config configs
