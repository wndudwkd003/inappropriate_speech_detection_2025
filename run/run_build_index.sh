#!/bin/bash


# PYTHONPATH 설정과 함께 실행
CUDA_VISIBLE_DEVICES=1 python -m scripts.rag.build_index --config configs
