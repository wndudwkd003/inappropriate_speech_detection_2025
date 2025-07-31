import os
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import EarlyStoppingCallback # TrainingArguments
from src.train.base_trainer import BaseTrainer
from dataclasses import asdict
from datasets import Dataset as HFDataset
from peft import PeftModel

# 맨 위 import 구역에 추가/교체
# try:
#     # TRL이 최상위에서 re-export하는 버전 (일부 버전에서만 동작)
#     from trl import DataCollatorForPreference
# except ImportError:
#     # 항상 동작하는 실제 경로
#     from trl.trainer.dpo_trainer import DataCollatorForPreference

# import torch

# def _embed_prehook(module, inputs):
#     (x,) = inputs  # x: [B, T]
#     V = module.num_embeddings
#     if x.min() < 0 or x.max() >= V:
#         bad = (x < 0) | (x >= V)
#         rows = torch.where(bad.any(dim=1))[0].tolist()
#         # 행별 min/max도 같이 출력
#         row_info = [(r, int(x[r].min().item()), int(x[r].max().item())) for r in rows]
#         print(f"[embed_prehook] out-of-range: V={V}, offenders(rows,min,max)={row_info}")
#         # 여기서 바로 막아 터뜨리면, 스택트레이스가 임베딩 직전에 멈춰 원인 파악이 쉽습니다.
#         raise AssertionError("input_ids contains out-of-range indices.")

class UnslothDPOTrainer(BaseTrainer):

    def setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.tokenizer_setup()
        self.tokenizer.padding_side = "left" # for dpo

        # # === [필수] 특수 토큰 정합화 + 임베딩 리사이즈 (LoRA 추가 전) ===
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.unk_token is None:
        #     # 실무에서 종종 eos로 대체, 문제없음
        #     self.tokenizer.unk_token = self.tokenizer.eos_token

        # # vocab_size(기본 어휘) 말고 실제 길이(len(tokenizer))로 비교해야 함
        # if self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
        #     self.model.resize_token_embeddings(len(self.tokenizer))

        # Vm = self.model.get_input_embeddings().num_embeddings
        # print(f"[check] len(tokenizer)={len(self.tokenizer)} "
        #     f"vocab_size={self.tokenizer.vocab_size} embed_size={Vm}")
        # for name in ["pad_token_id","eos_token_id","bos_token_id","unk_token_id"]:
        #     tid = getattr(self.tokenizer, name)
        #     if tid is not None:
        #         assert 0 <= tid < Vm, f"{name}={tid} out of range (<0 or >={Vm})"

        # DPO 전략: Pre-trained 모델에 SFT Adapter를 병합 -> DPO를 위해 새로운 Adapter를 병합된 모델에 추가
        sft_adapter_dir = os.path.join(
            self.cm.system.sft_model_for_dpo,
            self.cm.system.adapter_dir
        )

        if os.path.exists(sft_adapter_dir):
            print(f"Merging SFT adapter from {sft_adapter_dir}")
            # 1. SFT 어댑터 로드
            # self.model.load_adapter(sft_adapter_dir)

            # peft_model = PeftModel.from_pretrained(
            #     self.model,
            #     sft_adapter_dir,
            #     is_trainable=False,
            # )

            # merged_model = peft_model.merge_and_unload()
            # merged_dir = os.path.join(sft_adapter_dir, "merged")
            # os.makedirs(merged_dir, exist_ok=True)

            # merged_model.save_pretrained(merged_dir)
            # self.tokenizer.save_pretrained(merged_dir)
            # print(f"[DPO] SFT adapter merged & saved at: {merged_dir}")

            # # 2) merge된 가중치를 다시 불러와 DPO용 fresh LoRA를 붙인다
            # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            #     merged_dir,
            #     max_seq_length=self.cm.model.max_seq_length,
            #     dtype=self.cm.model.dtype,
            #     load_in_4bit=False,
            #     load_in_8bit=False,
            #     full_finetuning=False,
            # )

            # self.tokenizer_setup()
            print("SFT adapter merged successfully")

            """

            todo: 여기 기존의 모델이 가중치 수가 변하는지 확인 해야 함. 그리고 모델 불러오는 중간에 안쓰는 용량 차지하는게 아닌지 확인 해야 함

            """

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
            # use_gradient_checkpointing=True
        )

        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"Model embed_tokens size: {self.model.get_input_embeddings().num_embeddings}")
        print(f"Model setup complete. Total parameters: {self.model.num_parameters():,}")

        # self.model.get_input_embeddings().register_forward_pre_hook(_embed_prehook)


    def train(self, train_dataset: HFDataset, eval_dataset: HFDataset):
        # train_dataset = train_dataset.add_column("idx", list(range(len(train_dataset))))
        # if eval_dataset is not None:
        #     eval_dataset = eval_dataset.add_column("idx", list(range(len(eval_dataset))))

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

        # self.trainer = DPOTrainer(
        #     model=self.model,
        #     processing_class=self.tokenizer,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     callbacks=callbacks,
        #     args=training_args,
        #     data_collator=SafeDPOCollator(
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         vocab_size=self.model.get_input_embeddings().num_embeddings,
        #     ),
        # )

        # 사후 동기화(변화 없으면 noop)
        # if self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
        #     print("[warn] len(tokenizer)가 Trainer 생성 중 변했습니다. 임베딩 리사이즈 수행.")
        #     self.model.resize_token_embeddings(len(self.tokenizer))

        """
        RuntimeError: Triton Error [CUDA]: device-side assert triggered
        에러 발생하는거 해결해야 함

        """

        # from torch.utils.data import DataLoader
        # dl = self.trainer.get_train_dataloader()
        # first_batch = next(iter(dl))
        # emb = self.model.get_input_embeddings()

        # # 문제가 있다면 여기서 위의 콜레이터 어서션이 먼저 터지고, offenders가 출력됩니다.
        # for key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]:
        #     if key in first_batch:
        #         _ = emb(first_batch[key].to(self.model.device))


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



# class SafeDPOCollator(DataCollatorForPreference):
#     def __init__(self, *args, vocab_size: int, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._V = vocab_size

#     def __call__(self, features):
#         # features에는 방금 추가한 'idx'가 들어있습니다.
#         idxs = [f.get("idx", -1) for f in features]
#         batch = super().__call__(features)  # TRL 기본 콜레이트

#         V = self._V
#         # 각 샘플(row)별로 out-of-range 토큰이 있는지 검사
#         def check_matrix(name, x):
#             bad_rows = []
#             # x: [B, T]
#             if x is None:
#                 return
#             if not torch.is_floating_point(x):
#                 mn = int(x.min().item())
#                 mx = int(x.max().item())
#                 if mn < 0 or mx >= V:
#                     # 어느 샘플이 문제인지 찾기
#                     mask = (x < 0) | (x >= V)
#                     for r in torch.where(mask.any(dim=1))[0].tolist():
#                         bad_rows.append((idxs[r], int(x[r].min().item()), int(x[r].max().item())))
#                     raise AssertionError(
#                         f"[{name}] 토큰 ID 범위 오류: V={V}, offenders={bad_rows}"
#                     )

#         for k, x in batch.items():
#             if k.endswith("_input_ids"):
#                 check_matrix(k, x)

#         return batch
