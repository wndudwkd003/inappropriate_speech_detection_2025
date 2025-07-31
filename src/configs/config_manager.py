from src.configs.system_config import SystemConfig
from src.configs.model_config import ModelConfig
from src.configs.sft_config import SFTConfig
from src.configs.lora_config import LoRAConfig
from src.configs.dpo_config import DPOConfig
from src.configs.rag_config import RAGConfig
from src.configs.tune_config import TuneConfig
from src.data.prompt_manager import PromptVersion
from src.utils.path_utils import get_output_dir
import os, yaml, torch
from typing import TypeVar, Any
from dataclasses import is_dataclass, fields

T = TypeVar('T')


class ConfigManager:
    """싱글톤 패턴으로 구현된 설정 관리자"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.yaml_loader = YAMLLoader()
        self.type_converter = TypeConverter()

        # 로드된 설정들을 저장하는 딕셔너리
        self._configs = {}

        # 설정 파일 경로 저장
        self._config_paths = {}

        self._initialized = True

    def load_config(self, yaml_path: str, config_class: type[T], name: str = None) -> T:
        """설정을 로드하고 내부에 저장"""
        yaml_data = self.yaml_loader.load(yaml_path)
        config = self.type_converter.convert_to_dataclass(yaml_data, config_class)

        # 이름이 없으면 클래스 이름 사용
        if name is None:
            name = config_class.__name__.lower().replace('config', '')

        # 토큰이 비어있으면 저장된 토큰 로드
        if name == 'system' and hasattr(config, 'hf_token') and config.hf_token == "":
            token_path = "src/configs/tokens/token.yaml"
            if os.path.exists(token_path):
                token_data = self.yaml_loader.load(token_path)
                if 'hf_token' in token_data:
                    config.hf_token = token_data['hf_token']

        # 설정 저장
        self._configs[name] = config
        self._config_paths[name] = yaml_path

        return config

    def load_all_configs(self, config_dir: str = "configs"):
        """configs 디렉토리의 모든 설정 파일 로드"""


        # 설정 클래스 매핑
        config_mapping = {
            'system_config.yaml': (SystemConfig, 'system'),
            'model_config.yaml': (ModelConfig, 'model'),
            'sft_config.yaml': (SFTConfig, 'sft'),
            'lora_config.yaml': (LoRAConfig, 'lora'),
            'dpo_config.yaml': (DPOConfig, 'dpo'),
            'rag_config.yaml': (RAGConfig, 'rag'),
            'tune_config.yaml': (TuneConfig, 'tune')
        }

        for filename, (config_class, name) in config_mapping.items():
            yaml_path = os.path.join(config_dir, filename)
            if os.path.exists(yaml_path):
                self.load_config(yaml_path, config_class, name)

    def get_config(self, name: str) -> Any:
        """저장된 설정 반환"""
        if name not in self._configs:
            raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")
        return self._configs[name]

    def get_all_configs(self) -> dict[str, Any]:
        """모든 설정 반환"""
        return self._configs.copy()

    def save_config(self, yaml_path: str, name: str = None, config_obj: Any = None) -> None:
        """설정 저장"""
        if config_obj is None and name is not None:
            # 저장된 설정 사용
            if name not in self._configs:
                raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")
            config_obj = self._configs[name]
        elif config_obj is None:
            raise ValueError("저장할 설정 객체나 이름을 지정해주세요.")

        yaml_data = self.type_converter.convert_to_dict(config_obj)
        self.yaml_loader.save(yaml_path, yaml_data)

    def save_all_configs(self, output_dir: str):
        """모든 설정을 특정 디렉토리에 저장"""
        os.makedirs(output_dir, exist_ok=True)

        for name, config in self._configs.items():
            yaml_path = os.path.join(output_dir, f"{name}_config.yaml")
            self.save_config(yaml_path, config_obj=config)

    def update_config(self, name: str, updates: dict[str, Any]) -> Any:
        """특정 설정 업데이트"""
        if name not in self._configs:
            raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")

        config = self._configs[name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise KeyError(f"'{key}'는 {type(config).__name__}에 존재하지 않는 속성입니다.")

        return config

    def print_all_configs(self):
        # 예쁘게 출력
        for name, config in self._configs.items():
            print(f"{name.capitalize()} Configuration:")
            print(yaml.dump(self.type_converter.convert_to_dict(config), allow_unicode=True, default_flow_style=False))
            print("-" * 40)

    def reset(self):
        """모든 설정 초기화"""
        self._configs.clear()
        self._config_paths.clear()

    @property
    def system(self) -> SystemConfig:
        """시스템 설정 빠른 접근"""
        return self.get_config('system')

    @property
    def model(self) -> ModelConfig:
        """모델 설정 빠른 접근"""
        return self.get_config('model')

    @property
    def sft(self) -> SFTConfig:
        """SFT 설정 빠른 접근"""
        return self.get_config('sft')

    @property
    def lora(self) -> LoRAConfig:
        """LoRA 설정 빠른 접근"""
        return self.get_config('lora')

    @property
    def dpo(self) -> DPOConfig:
        """DPO 설정 빠른 접근"""
        return self.get_config('dpo')

    @property
    def rag(self) -> RAGConfig:
        """RAG 설정 빠른 접근"""
        return self.get_config('rag')

    @property
    def tune(self) -> TuneConfig:
        """튜닝 설정 빠른 접근"""
        return self.get_config('tune')

class YAMLLoader:
    def load(self, yaml_path: str) -> dict[str, Any]:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def save(self, yaml_path: str, data: dict[str, Any]) -> None:
        # 폴더 존재하는지 확인
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, allow_unicode=True, default_flow_style=False)


class TypeConverter:
    def __init__(self):
        # dtype 문자열 -> torch dtype 매핑
        self.dtype_map = dict(
            fp16=torch.float16,
            bf16=torch.bfloat16,
            fp32=torch.float32,
            nf4="nf4", # 특수 타입 for qlora
        )

        self.enum_map = {
            'prompt_version': PromptVersion,
            # 필요하면 다른 Enum들도 추가
        }

    def convert_to_dataclass(self, yaml_data: dict[str, Any], target_class: type[T]) -> T:
        # YAML 데이터를 dataclass 인스턴스로 변환
        converted_data = yaml_data.copy()

        # dtype 변환
        if 'dtype' in converted_data and isinstance(converted_data['dtype'], str):
            converted_data['dtype'] = self.dtype_map.get(converted_data['dtype'], converted_data['dtype'])

        # Enum 변환
        for field_name, enum_class in self.enum_map.items():
            if field_name in converted_data and isinstance(converted_data[field_name], str):
                # 문자열 값으로 Enum 찾기
                for enum_member in enum_class:
                    if enum_member.value == converted_data[field_name]:
                        converted_data[field_name] = enum_member
                        break

        # ===== 여기 추가 =====
        # dataclass 필드 이름 가져오기
        valid_fields = {f.name for f in fields(target_class)}

        # 유효한 필드만 필터링
        filtered_data = {}
        for key, value in converted_data.items():
            if key in valid_fields:
                filtered_data[key] = value
            else:
                print(f"경고: '{key}' 필드는 {target_class.__name__}에 존재하지 않아 무시됩니다.")

        return target_class(**filtered_data)  # converted_data 대신 filtered_data 사용

    def convert_to_dict(self, obj: Any) -> dict[str, Any]:
        if not is_dataclass(obj):
            raise TypeError(f"데이터세트 클래스가 아닙니다. {type(obj)}")

        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)

            # dtype 변환
            if isinstance(value, torch.dtype):
                for dtype_str, dtype_val in self.dtype_map.items():
                    if dtype_val == value:
                        value = dtype_str
                        break

            # Enum 변환 추가!
            elif hasattr(value, 'value') and hasattr(value, '__class__'):
                # Enum인지 확인하고 value 속성으로 변환
                if value.__class__ in self.enum_map.values():
                    value = value.value

            result[field.name] = value

        return result

def init_config_manager(dir: str = "configs", train_type: str = "dpo", is_tune: bool = False) -> ConfigManager:
    config_manager = ConfigManager()

    config_dir = dir
    if train_type == "dpo":
        config_dir = os.path.join(config_dir, "configs")

    config_manager.load_all_configs(config_dir=config_dir)

    if train_type == "dpo":
        config_manager.update_config("system", {"sft_model_for_dpo": dir})
        base_path = config_manager.dpo.output_dir
    elif train_type == "sft":
        base_path = config_manager.sft.output_dir

    # # RAG 설정을 켰는데 데이터 디렉토리가 rag_results가 아닌경우 자동으로 설정
    # if config_manager.rag.use_rag and os.path.basename(config_manager.system.data_raw_dir) != "rag_results":
    #     print("RAG 설정이 켜져있습니다. 데이터 디렉토리를 자동으로 변경합니다.")
    #     config_manager.update_config("system", {"data_raw_dir": "data/rag_results"})
    #     print(f"Current data_raw_dir: {config_manager.system.data_raw_dir}")

    essential = dict(
        model_id = config_manager.model.model_id,
        r=config_manager.lora.r,
        lora_alpha=config_manager.lora.lora_alpha,
        lora_dropout=config_manager.lora.lora_dropout,
        dtype=config_manager.model.dtype,
    )

    output_dir = get_output_dir(base_path=base_path, essential=essential, train_type=train_type)
    if not is_tune:
        os.makedirs(output_dir, exist_ok=True)

    fp16 = config_manager.model.dtype == torch.float16
    bf16 = config_manager.model.dtype == torch.bfloat16

    config_manager.update_config(train_type, {
        "output_dir": output_dir, "logging_dir": os.path.join(output_dir, "logs"),
        "fp16": fp16, "bf16": bf16,
    })
    config_manager.print_all_configs()
    return config_manager
