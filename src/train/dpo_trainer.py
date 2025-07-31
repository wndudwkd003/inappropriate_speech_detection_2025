import os
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import EarlyStoppingCallback # TrainingArguments
from src.train.base_trainer import BaseTrainer, print_model_parameters
from dataclasses import asdict
from datasets import Dataset as HFDataset
from peft import PeftModel

class UnslothDPOTrainer(BaseTrainer):
    def setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
            device_map="auto",
        )
        print_model_parameters(self.model, label="Base Model")

        self.tokenizer_setup()
        self.tokenizer.padding_side = "left" # for dpo

        # DPO 전략: Pre-trained 모델에 SFT Adapter를 병합 -> DPO를 위해 새로운 Adapter를 병합된 모델에 추가
        sft_adapter_dir = os.path.join(
            self.cm.system.sft_model_for_dpo,
            self.cm.system.adapter_dir
        )

        if os.path.exists(sft_adapter_dir):
            print(f"Merging SFT adapter from {sft_adapter_dir}")
            # SFT 어댑터 로드
            peft_model = PeftModel.from_pretrained(
                self.model,
                sft_adapter_dir,
                is_trainable=False,
            )

            print_model_parameters(peft_model, label="Merged SFT Model 1")
            merged_model = peft_model.merge_and_unload()
            self.model = merged_model
            print_model_parameters(self.model, label="Merged SFT Model 2")
            print("SFT adapter merged successfully")

        # DPO를 위한 새로운 LoRA 어댑터 추가
        print("Adding new LoRA adapter for DPO training")
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

        print_model_parameters(self.model, label="Merged DPO Model")

        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"Model embed_tokens size: {self.model.get_input_embeddings().num_embeddings}")
        print(f"Model setup complete. Total parameters: {self.model.num_parameters():,}")


    def train(self, train_dataset: HFDataset, eval_dataset: HFDataset):
        dpo_dict = asdict(self.cm.dpo)
        dpo_dict.update({
            "padding_value": int(self.tokenizer.pad_token_id),   # <-- 중요
            "label_pad_token_id": -100,                          # labels용은 -100 유지
        })
        training_args = DPOConfig(**dpo_dict)

        callbacks = [EarlyStoppingCallback(
            early_stopping_patience=self.cm.model.early_stopping,
            early_stopping_threshold=self.cm.model.early_stopping_threshold
        )]if self.cm.model.early_stopping else None


        self.trainer = DPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            args=training_args,
        )

        trainer_stats = self.trainer.train()
        return trainer_stats

    def save_adapter(self, save_path:str|None = None):
        """LoRA 어댑터 저장"""
        if save_path is None:
            save_path = os.path.join(self.cm.dpo.output_dir, self.cm.system.adapter_dir)

        self.model.save_pretrained(save_path)
        # self.tokenizer.save_pretrained(save_path)

        # 설정 파일도 함께 저장
        self.cm.update_config("system", {"hf_token": ""}) # remove token for security
        self.cm.save_all_configs(os.path.join(self.cm.dpo.output_dir, "configs"))

        print(f"Adapter saved to: {save_path}")
        return save_path

