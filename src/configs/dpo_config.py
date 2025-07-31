from dataclasses import dataclass, field

@dataclass
class DPOConfig:
    output_dir: str = "output" # it will be changed by the function
    num_train_epochs: int = 5
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 1
    gradient_accumulation_steps: int = 1 # GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    eval_strategy: str = "steps" # "no", "epoch", "steps"
    save_strategy: str = "steps" # "no", "epoch", "steps"
    eval_steps: int | None = 100 # 100 613
    save_steps: int | None = 100 # 100 613
    logging_steps: int = 25
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    save_total_limit: int = 1
    logging_dir: str = "logs"
    report_to: list[str] | None = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = True
    bf16: bool = False
    # packing: bool = False
    # gradient_checkpointing: bool = True
    # activation_offloading: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "adamw_torch" # "adamw_torch" is default, adamw_hf or "adamw_8bit" or "paged_adamw_8bit"
    seed: int = 42
    push_to_hub: bool = False
    padding_value: int = -100  # padding value for loss calculation

    # dpo parameters
    # padding_value: int = "tokenizer.pad_token_id"
    # model_init_kwargs: dict = field(default_factory=dict)
    # ref_model_init_kwargs: dict | None = None

    # dpo_model_path: str = "output/2025-07-24_20-06-25_kakaocorp_kanana-1.5-8b-base_r_128_ra_256_rd_0_sft"
    # beta: float = 0.1
    # loss_type: str = "sigmoid"
    # label_smoothing: float = 0.0
    # reference_free: bool = False
    # precompute_ref_log_probs: bool = True
    # max_length: int = 2048
    # max_prompt_length: int = 1024
    # max_target_length: int = 512
