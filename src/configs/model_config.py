from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    model_id: str = ""
    dtype: str|torch.dtype = ""
    max_seq_length: int = 4096 # 4096 2048

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    full_finetune: bool = False

    # only train
    early_stopping: int = 3
    early_stopping_threshold: float = 0.001

    # only inference
    max_new_tokens: int = 512
    do_sample: bool = False
