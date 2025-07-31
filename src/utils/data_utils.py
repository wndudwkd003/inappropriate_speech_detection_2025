import os
from typing import TypeVar
from datasets import Dataset as HFDataset          # ★ 추가
from src.configs.config_manager import ConfigManager
from src.data.dataset_factory import DatasetFactory
from src.data.dpo_dataset import DPODataset
from datasets import Dataset as HFDataset

T = TypeVar("T")

def convert_hf(ds):
    return HFDataset.from_list(list(ds))

def prepare_dataset(config_manager: ConfigManager, tokenizer, task_type: str = "sft"):
    prompt_version = config_manager.system.prompt_version
    print("Current prompt version:", prompt_version)

    # 공통 인자 정리
    common_args = dict(
        dataset_type=task_type,
        tokenizer=tokenizer,
        config_manager=config_manager,
    )

    train_dataset = DatasetFactory.create_dataset(
        fname=os.path.join(config_manager.system.data_raw_dir, "train.json"),
        data_shuffle=config_manager.system.data_shuffle,
        **common_args
    )

    eval_dataset = DatasetFactory.create_dataset(
        fname=os.path.join(config_manager.system.data_raw_dir, "dev.json"),
        data_shuffle=False, # 평가 데이터는 셔플하지 않음
        **common_args
    )

    if task_type.lower() == "dpo":
        print("Converting DPO dataset to Hugging Face format...")
        train_dataset = convert_hf(train_dataset)
        eval_dataset = convert_hf(eval_dataset)

    return train_dataset, eval_dataset


def prepare_test_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    test_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "test.json"),
        tokenizer=tokenizer
    )

    return test_dataset

