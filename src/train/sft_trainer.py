from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
from src.train.base_trainer import BaseTrainer
from src.data.datasets.sft_dataset import SFTDataset
from src.data.datasets.base_dataset import DataCollatorForSupervisedDataset
from transformers import AutoModelForSequenceClassification
from dataclasses import asdict
import os
from src.utils.metric_utils import TrainResult
from transformers import DataCollatorWithPadding

class UnslothSFTTrainer(BaseTrainer):
    def setup_model(self):
        add_args = dict()

        if self.cm.model.num_classes != -1:
            add_args["num_labels"] = self.cm.model.num_classes

        if not self.cm.system.only_decode:
            add_args["auto_model"] = AutoModelForSequenceClassification
            fast_model = FastModel
            print("Using FastModel for classification training.")
        else:
            fast_model = FastLanguageModel
            print("Using FastLanguageModel for generation training.")

        print(f"model_id: {self.cm.model.model_id}")
        self.model, self.tokenizer = fast_model.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype if self.cm.system.only_decode else None,
            load_in_4bit=self.cm.model.load_in_4bit,
            load_in_8bit=self.cm.model.load_in_8bit,
            full_finetuning=self.cm.model.full_finetune,
            trust_remote_code=True,
            **add_args
        )

        print("model parameters:" + str(sum(p.numel() for p in self.model.parameters())))

        self.tokenizer_setup()
        self.tokenizer.padding_side = "right"

        # LoRA 설정 (생성형 모델에서만)
        if not self.cm.model.full_finetune and self.cm.system.only_decode:
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
            # "ddp_find_unused_parameters": False,
            # "dataloader_pin_memory": False,
            # "dataloader_num_workers": 0,
            # "remove_unused_columns": False,
        })

        training_args = TrainingArguments(**sft_dict)

        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
        )

        callbacks = [EarlyStoppingCallback(
            early_stopping_patience=self.cm.model.early_stopping,
            early_stopping_threshold=self.cm.model.early_stopping_threshold
        )] if self.cm.model.early_stopping not in [-1, None] else None

        # 생성형 모델인지 분류 모델인지에 따라 다른 Trainer 사용
        if self.cm.system.only_decode:
            # 생성형 모델: SFTTrainer 사용
            print("Using SFTTrainer for generation tasks.")
            self.trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                args=training_args,
            )
        else:
            # 분류 모델: 일반 Trainer 사용
            print("Using standard Trainer for classification tasks.")

            classification_data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt"
            )

            self.trainer = Trainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=classification_data_collator,
                callbacks=callbacks,
                args=training_args,
            )

        trainer_stats = self.trainer.train()
        result = TrainResult(trainer_stats, self.trainer.state.log_history)

        print("=== Training completed ===")
        print(f"Final train loss: {result.training_loss:.4f}")
        print(f"Total steps: {result.global_step}")
        print(f"Log history entries: {len(result.log_history)}")

        return result

    def save_adapter(self, save_path: str | None = None):
        """LoRA 어댑터 또는 전체 모델 저장"""
        if save_path is None:
            save_path = os.path.join(self.cm.sft.output_dir, self.cm.system.adapter_dir)

        # 모델 저장
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            # fallback: torch save
            import torch
            torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

        # 토크나이저 저장 (full finetuning이거나 분류 모델인 경우)
        if self.cm.model.full_finetune or not self.cm.system.only_decode:
            self.tokenizer.save_pretrained(save_path)

        # 설정 파일도 함께 저장
        self.cm.update_config("system", {"hf_token": ""})  # remove token for security
        self.cm.save_all_configs(os.path.join(self.cm.sft.output_dir, "configs"))

        print(f"Model saved to: {save_path}")
        return save_path
