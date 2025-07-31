import os
from dataclasses import asdict
from datasets import Dataset

from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)

from src.utils.data_utils import prepare_dataset
from src.configs.config_manager import ConfigManager

def cache_dataset_for_opensloth(config_manager: ConfigManager, task_type: str, force_cache: bool = False) -> str:
    """기존 SFTDataset을 OpenSloth용 HuggingFace Dataset으로 캐싱"""



    cache_dir = os.path.join("data", f"cache_{task_type}_dataset")

    # 캐시가 이미 있고 force_cache가 False면 건너뛰기
    if os.path.exists(cache_dir) and not force_cache:
        print(f"Using cached dataset at {cache_dir}")
        return cache_dir

    # 캐시가 없으면 에러 - 별도로 캐싱 필요
    raise FileNotFoundError(f"Dataset cache not found at {cache_dir}. Please run caching script first.")



def config_to_opensloth(config_manager: ConfigManager, data_path: str, devices: list):
    """ConfigManager를 OpenSloth 설정으로 변환 (모델 병렬화)"""

    # 1. OpenSloth 설정 생성 (모델 병렬화)
    opensloth_config = OpenSlothConfig(
        data_cache_path=data_path,
        devices=devices,
        fast_model_args=FastModelArgs(
            model_name=config_manager.model.model_id,
            max_seq_length=config_manager.model.max_seq_length,
            dtype=config_manager.model.dtype,
            load_in_4bit=True if config_manager.lora.use_qlora else False,
            load_in_8bit=False,
            trust_remote_code=True,
            # 모델 병렬화를 위한 추가 설정
            device_map="balanced",  # 자동 디바이스 분할
            # max_memory={i: "20GB" for i in devices},  # 각 GPU당 최대 메모리
            # low_cpu_mem_usage=True,  # CPU 메모리 사용량 최소화
        ),
        lora_args=LoraArgs(
            r=config_manager.lora.r,
            lora_alpha=config_manager.lora.lora_alpha,
            target_modules=config_manager.lora.target_modules,
            lora_dropout=config_manager.lora.lora_dropout,
            bias=config_manager.lora.bias,
            use_rslora=getattr(config_manager.lora, 'use_rslora', False),
        ),
        # 모델 병렬화를 위한 추가 설정
        model_parallel=True,  # 모델 병렬화 활성화
        tensor_parallel_size=len(devices),  # 텐서 병렬화 크기
    )

    # 2. 훈련 설정 변환 (모델 병렬화 고려)
    sft_dict = asdict(config_manager.sft)

    if isinstance(sft_dict.get("report_to"), list):
        report_to_list = sft_dict["report_to"]
        if report_to_list:
            sft_dict["report_to"] = report_to_list[0]
        else:
            sft_dict["report_to"] = "none"

    # 모델 병렬화를 위한 배치 크기 조정
    # 모델이 분할되므로 각 디바이스의 배치 크기는 더 클 수 있음
    sft_dict.update({
        "data_seed": config_manager.system.seed,
        "gradient_accumulation_steps": 1,  # 모델 병렬화에서는 일반적으로 1
        "ddp_find_unused_parameters": False,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        # 모델 병렬화를 위한 추가 설정
        "gradient_checkpointing": True,  # 메모리 효율성
        "fp16": True,  # 메모리 절약
        "deepspeed": None,  # DeepSpeed 비활성화 (OpenSloth 자체 병렬화 사용)
    })

    training_config = TrainingArguments(**sft_dict)

    return opensloth_config, training_config
