import unsloth
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from src.train.sft_trainer import UnslothSFTTrainer
from src.utils.data_utils import prepare_dataset
from src.utils.huggingface_utils import init_hub_env
from src.utils.metric_utils import save_metrics
from src.configs.config_manager import ConfigManager, init_config_manager
import argparse

CURRENT_TRAIN_TYPE = "sft"



def main(config_manager: ConfigManager):
    # SFT 트레이너 초기화
    trainer = UnslothSFTTrainer(config_manager)

    train_dataset, eval_dataset = prepare_dataset(
        config_manager=config_manager,
        tokenizer=trainer.tokenizer,
        task_type=CURRENT_TRAIN_TYPE,
    )

    metrics = trainer.train(train_dataset, eval_dataset)
    trainer.save_adapter()

    save_metrics(metrics, config_manager.sft.output_dir)
    print(f"Training completed. saved at {config_manager.sft.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT Model")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager(dir=args.config, train_type=CURRENT_TRAIN_TYPE)
    config_manager.update_config(CURRENT_TRAIN_TYPE, {"seed": config_manager.system.seed})
    init_hub_env(config_manager.system.hf_token)

    set_seed(config_manager.system.seed, deterministic=config_manager.system.deterministic)
    # set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    # 메인 함수 실행
    main(config_manager)
