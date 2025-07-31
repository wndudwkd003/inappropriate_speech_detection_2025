from abc import ABC, abstractmethod
from src.configs.config_manager import ConfigManager
import os

class BaseTrainer(ABC):
    def __init__(self, config_manager: ConfigManager):
        self.cm = config_manager
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.setup_model()

    # 토크나이저 설정
    def tokenizer_setup(self):
        # if tokenizer pad token is None, set it to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Pad token이 설정되지 않아 eos_token으로 대체했습니다.")

        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = self.tokenizer.eos_token
            print("UNK token이 설정되지 않아 eos_token으로 대체했습니다.")

        # if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
        #     self.model.resize_token_embeddings(len(self.tokenizer))


    @abstractmethod
    def setup_model(self):
        """모델과 토크나이저를 설정하는 메서드"""
        pass


    @abstractmethod
    def train(self, train_dataset, eval_dataset):
        """훈련을 수행하는 메서드"""
        pass

    @abstractmethod
    def save_adapter(self, save_path: str | None = None):
        """LoRA 어댑터를 저장하는 메서드"""
        pass




def print_model_parameters(model, label=""):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[{label}] Total Parameters: {total_params:,}")
    print(f"[{label}] Trainable Parameters: {trainable_params:,}")
