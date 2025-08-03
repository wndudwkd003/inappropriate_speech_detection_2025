from dataclasses import dataclass, field
from typing import Any
from ray import tune

@dataclass
class TuneConfig:
    output_dir: str = "./outputs/tune_results"
    local_dir: str = "./ray_results"
    name: str = "llm_hyperparameter_tuning"

    num_samples: int = 50  # 총 실험 횟수
    max_concurrent_trials: int = 1  # 동시 실행할 트라이얼 수


    scheduler_type: str = "ASHA"  # "ASHA", "PBT", "BOHB", "MedianStopping"

    # ASHA 스케줄러 설정
    asha_config: dict[str, Any] = field(default_factory=lambda: {
        "metric": "eval_loss",
        "mode": "min",
        "max_t": 2000,  # 2000스텝으로 설정
        "grace_period": 200,  # max_t의 10% 정도
        "reduction_factor": 3,
        "brackets": 1
    })



    # 학습률 관련
    learning_rate: dict[str, Any] = field(default_factory=lambda: {
        "type": "loguniform",
        "low": 1e-6,
        "high": 1e-3
    })

    warmup_ratio: dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.01,
        "high": 0.1
    })

    weight_decay: dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.001,
        "high": 0.1
    })

    lr_scheduler_type: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": ["cosine", "cosine_with_restarts", "polynomial"] # "linear"
    })

    # 배치 크기 관련
    per_device_train_batch_size: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [1, 2]
    })

    gradient_accumulation_steps: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [1, 2, 4, 8, 16, 32]  # GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    })

    # LoRA 관련 파라미터
    lora_r: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [8, 16, 32, 64, 128]
    })

    lora_alpha: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [8, 16, 32, 64, 128, 256]
    })

    lora_dropout: dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.0,
        "high": 0.3
    })

    # 기존 fixed_params에서 target_modules 제거하고
    lora_target_modules: dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [
            ["q_proj", "v_proj"],  # 기본
            ["q_proj", "k_proj", "v_proj", "o_proj"],  # attention만
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 전체
            ["gate_proj", "up_proj", "down_proj"]  # MLP만
        ]
    })

    # 체크포인트 설정
    checkpoint_config: dict[str, Any] = field(default_factory=lambda: {
        "checkpoint_score_attr": "eval_loss",
        "checkpoint_freq": 1,
        "num_to_keep": 3
    })

    progress_reporter: dict[str, Any] = field(default_factory=lambda: {
        "metric_columns": ["eval_loss", "eval_accuracy", "training_iteration"],
        "max_progress_rows": 20,
        "max_error_rows": 5,
        "max_column_length": 20
    })

    best_model_selection: dict[str, Any] = field(default_factory=lambda: {
        "metric": "eval_loss",
        "mode": "min",
        "scope": "all"  # "all", "last", "avg"
    })


    def to_search_space(self) -> dict[str, Any]:
        """TuneConfig의 파라미터들을 Ray Tune search space로 변환"""
        search_space = {}

        # learning_rate
        if self.learning_rate["type"] == "loguniform":
            search_space["learning_rate"] = tune.loguniform(
                self.learning_rate["low"],
                self.learning_rate["high"]
            )
        elif self.learning_rate["type"] == "uniform":
            search_space["learning_rate"] = tune.uniform(
                self.learning_rate["low"],
                self.learning_rate["high"]
            )
        elif self.learning_rate["type"] == "choice":
            search_space["learning_rate"] = tune.choice(self.learning_rate["values"])

        # warmup_ratio
        if self.warmup_ratio["type"] == "uniform":
            search_space["warmup_ratio"] = tune.uniform(
                self.warmup_ratio["low"],
                self.warmup_ratio["high"]
            )
        elif self.warmup_ratio["type"] == "choice":
            search_space["warmup_ratio"] = tune.choice(self.warmup_ratio["values"])

        # weight_decay
        if self.weight_decay["type"] == "uniform":
            search_space["weight_decay"] = tune.uniform(
                self.weight_decay["low"],
                self.weight_decay["high"]
            )
        elif self.weight_decay["type"] == "choice":
            search_space["weight_decay"] = tune.choice(self.weight_decay["values"])

        # lr_scheduler_type
        search_space["lr_scheduler_type"] = tune.choice(self.lr_scheduler_type["values"])

        # per_device_train_batch_size
        search_space["per_device_train_batch_size"] = tune.choice(
            self.per_device_train_batch_size["values"]
        )

        # gradient_accumulation_steps
        search_space["gradient_accumulation_steps"] = tune.choice(
            self.gradient_accumulation_steps["values"]
        )

        # lora_r
        search_space["lora_r"] = tune.choice(self.lora_r["values"])

        # lora_alpha
        search_space["lora_alpha"] = tune.choice(self.lora_alpha["values"])

        # lora_dropout
        if self.lora_dropout["type"] == "uniform":
            search_space["lora_dropout"] = tune.uniform(
                self.lora_dropout["low"],
                self.lora_dropout["high"]
            )
        elif self.lora_dropout["type"] == "choice":
            search_space["lora_dropout"] = tune.choice(self.lora_dropout["values"])

        # lora_target_modules
        search_space["lora_target_modules"] = tune.choice(self.lora_target_modules["values"])

        return search_space
