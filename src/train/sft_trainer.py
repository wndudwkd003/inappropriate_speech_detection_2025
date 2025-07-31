from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.train.base_trainer import BaseTrainer
from src.data.sft_dataset import SFTDataset
from src.data.dataset import DataCollatorForSupervisedDataset
from dataclasses import asdict
import os

class UnslothSFTTrainer(BaseTrainer):
    def setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=self.cm.model.load_in_4bit,
            load_in_8bit=self.cm.model.load_in_8bit,
            full_finetuning=self.cm.model.full_finetune,
            # device_map="balanced",
            trust_remote_code=True,
        )

        self.tokenizer_setup()
        self.tokenizer.padding_side = "right"

        if not self.cm.model.full_finetune:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.cm.lora.r,
                target_modules=self.cm.lora.target_modules,
                lora_alpha=self.cm.lora.lora_alpha,
                lora_dropout=self.cm.lora.lora_dropout,
                bias=self.cm.lora.bias,
                random_state=self.cm.system.seed,
                init_lora_weights=self.cm.lora.init_lora_weights,
            )

    def train(self, train_dataset: SFTDataset, eval_dataset: SFTDataset):
        sft_dict = asdict(self.cm.sft)
        sft_dict.update({
            # "data_seed": self.cm.system.seed,
            # "ddp_find_unused_parameters": False,"ddp_find_unused_parameters": False,
            # "dataloader_pin_memory": False,  # 이 옵션 추가
            # "dataloader_num_workers": 0,     # 이 옵션도 추가 (안전을 위해)
            # "remove_unused_columns": False,  # 이 옵션 추가
        })

        training_args = TrainingArguments(**sft_dict)

        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
            # model=self.model,
        )

        callbacks = [EarlyStoppingCallback(
            early_stopping_patience=self.cm.model.early_stopping,
            early_stopping_threshold=self.cm.model.early_stopping_threshold
        )]if self.cm.model.early_stopping else None

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            args=training_args,
        )

        trainer_stats = self.trainer.train()
        return trainer_stats

    def save_adapter(self, save_path:str|None = None):
        """LoRA 어댑터 저장"""
        if save_path is None:
            save_path = os.path.join(self.cm.sft.output_dir, self.cm.system.adapter_dir)

        self.model.save_pretrained(save_path)
        if self.cm.model.full_finetune:
            self.tokenizer.save_pretrained(save_path)

        # self.tokenizer.save_pretrained(save_path)

        # 설정 파일도 함께 저장
        self.cm.update_config("system", {"hf_token": ""}) # remove token for security
        self.cm.save_all_configs(os.path.join(self.cm.sft.output_dir, "configs"))

        print(f"Adapter saved to: {save_path}")
        return save_path

