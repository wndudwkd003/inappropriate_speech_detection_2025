from dataclasses import dataclass, field
from src.data.prompt_manager import PromptVersion

@dataclass
class SystemConfig:
    is_train: bool = True
    additional_info: str = "" # it will be concatenated to other configs
    seed: int = 42
    deterministic: bool = True
    hf_token: str = "" # it will be set by the function
    use_lora: bool = True
    use_qlora: bool = False
    data_raw_dir: str = "data/raw"
    data_rag_dir: str = "data/rag"
    data_hangul_info: bool = True

    prompt_version: PromptVersion = PromptVersion.V1
    data_question_length_limit: int = 500
    data_shuffle: bool = False

    # dpo 관련
    sft_model_for_dpo: str = ""

    # only inference
    save_dir: str = "" # if will be set by the function
    adapter_dir: str = "lora_adapter"
    test_result_dir: str = "test_results" # it will be set by the function

    is_cot: bool = False
