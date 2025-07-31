import argparse
import os
from transformers import set_seed

from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

from src.utils.seed_utils import set_all_seeds
from src.utils.huggingface_utils import init_hub_env
from src.utils.metric_utils import save_metrics
from src.configs.config_manager import ConfigManager, init_config_manager
from src.utils.opensloth_utils import cache_dataset_for_opensloth, config_to_opensloth

from src.train.sft_trainer import UnslothSFTTrainer
from src.utils.data_utils import prepare_dataset

CURRENT_TRAIN_TYPE = "sft"





def main(config_manager: ConfigManager, devices: list, force_cache: bool):

    # trainer = UnslothSFTTrainer(config_manager)

    # train_dataset, eval_dataset = prepare_dataset(
    #     config_manager=config_manager,
    #     tokenizer=trainer.tokenizer,
    #     task_type="dpo",
    # )




    # 2. OpenSloth 설정 변환
    opensloth_config, training_config = config_to_opensloth(config_manager, config_manager.system.data_raw_dir, devices)

    # 3. 디버그 정보 출력
    devices = opensloth_config.devices
    batch_size = training_config.per_device_train_batch_size
    grad_accum = training_config.gradient_accumulation_steps

    print(f"→ Devices: {devices}")
    print(f"→ Global batch size: {len(devices) * batch_size * grad_accum}")
    print(f"→ Grad accumulation: {grad_accum}")

    # 4. OpenSloth 훈련 실행
    setup_envs(opensloth_config, training_config)
    metrics = run_mp_training(devices, opensloth_config, training_config)

    # 5. 메트릭 저장
    save_metrics(metrics, config_manager.sft.output_dir)
    print(f"Training completed. saved at {config_manager.sft.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT Model with OpenSloth")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1], help="GPU devices to use")
    parser.add_argument("--force_cache", action="store_true", help="Force re-cache dataset")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager(dir=args.config, train_type=CURRENT_TRAIN_TYPE)
    config_manager.update_config(CURRENT_TRAIN_TYPE, {"seed": config_manager.system.seed})

    # Multi-GPU 설정 (하드코딩)
    devices = args.devices
    force_cache = args.force_cache

    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    # 메인 함수 실행
    main(config_manager, devices, force_cache)
