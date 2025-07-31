# src/tune/trainer/sft_tune_trainer.py
from src.train.sft_trainer import UnslothSFTTrainer
from src.data.sft_dataset import SFTDataset
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data.dataset import DataCollatorForSupervisedDataset
from trl import SFTTrainer
from dataclasses import asdict
from ray import tune
import copy

class UnslothSFTTuneTrainer(UnslothSFTTrainer):
    def __init__(self, config_manager, tune_config):
        super().__init__(config_manager)
        self.tune_config = tune_config

    def tune_train(self, train_dataset: SFTDataset, eval_dataset: SFTDataset, tuned_config: dict):
        """Ray Tune에서 호출되는 학습 함수"""
        try:
            # 1. 튜닝된 파라미터로 TrainingArguments 생성
            training_args_dict = self._create_training_args_dict(tuned_config)
            training_args = TrainingArguments(**training_args_dict)

            # 2. LoRA 설정은 적용하지 않음 (모델 재설정 안함)
            # self._apply_lora_config(tuned_config)  # 주석 처리

            # 3. 데이터 콜레이터 생성
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

            # 4. 콜백 설정
            callbacks = [EarlyStoppingCallback(
                early_stopping_patience=self.cm.model.early_stopping,
                early_stopping_threshold=self.cm.model.early_stopping_threshold
            )] if self.cm.model.early_stopping else None

            # 5. SFTTrainer 생성 및 학습
            trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                args=training_args,
            )

            # 6. 학습 실행
            trainer_stats = trainer.train()

            # 7. 평가 결과 반환
            eval_results = trainer.evaluate()

            return {
                'eval_loss': eval_results.get('eval_loss', float('inf')),
                'eval_accuracy': eval_results.get('eval_accuracy', 0.0),
                'trainer_stats': trainer_stats
            }

        except Exception as e:
            print(f"Error in tune_train: {e}")
            return {
                'eval_loss': float('inf'),
                'eval_accuracy': 0.0,
                'trainer_stats': None
            }

    def _create_training_args_dict(self, tuned_config: dict) -> dict:
        """튜닝된 파라미터로 TrainingArguments용 딕셔너리 생성"""
        # 기본 SFT 설정을 딕셔너리로 변환
        sft_dict = asdict(self.cm.sft)
        sft_dict.update({
            "data_seed": self.cm.system.seed,
        })

        # 튜닝된 하이퍼파라미터로 덮어쓰기
        tuning_param_mapping = {
            'learning_rate': 'learning_rate',
            'warmup_ratio': 'warmup_ratio',
            'weight_decay': 'weight_decay',
            'lr_scheduler_type': 'lr_scheduler_type',
            'per_device_train_batch_size': 'per_device_train_batch_size',
            'gradient_accumulation_steps': 'gradient_accumulation_steps'
        }

        for tuned_key, sft_key in tuning_param_mapping.items():
            if tuned_key in tuned_config:
                sft_dict[sft_key] = tuned_config[tuned_key]

        return sft_dict

    def _apply_lora_config(self, tuned_config: dict):
        """LoRA 관련 설정 적용"""
        lora_params_changed = any(key in tuned_config for key in
                                ['lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules'])

        if lora_params_changed:
            # 원본 설정 백업
            original_lora_config = copy.deepcopy(self.cm.lora)

            # 임시로 LoRA 설정 업데이트
            lora_updates = {}
            if 'lora_r' in tuned_config:
                lora_updates['r'] = tuned_config['lora_r']
            if 'lora_alpha' in tuned_config:
                lora_updates['lora_alpha'] = tuned_config['lora_alpha']
            if 'lora_dropout' in tuned_config:
                lora_updates['lora_dropout'] = tuned_config['lora_dropout']
            if 'lora_target_modules' in tuned_config:
                lora_updates['target_modules'] = tuned_config['lora_target_modules']

            # ConfigManager 임시 업데이트
            self.cm.update_config('lora', lora_updates)

            try:
                # 기존 모델 해제
                if hasattr(self, 'model'):
                    del self.model

                # GPU 메모리 정리
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 모델 재설정
                self.setup_model()

            except Exception as e:
                print(f"모델 재설정 실패: {e}")
                # 원본 설정 복원
                self.cm._configs['lora'] = original_lora_config
                raise e

            # 원본 설정 복원
            self.cm._configs['lora'] = original_lora_config

    def _restore_model_if_needed(self):
        """필요시 모델 복원 (현재는 사용하지 않음)"""
        pass
