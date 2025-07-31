# src/tune/tune_objective.py
from src.tune.trainer.sft_tune_trainer import UnslothSFTTuneTrainer
from src.utils.data_utils import prepare_dataset
from ray import tune

def get_tune_trainer(train_type: str, config_manager, tune_config):
    """팩토리 메서드: 학습 타입에 따라 적절한 튜닝 트레이너 반환"""
    trainers = {
        "sft": UnslothSFTTuneTrainer,
    }
    if train_type not in trainers:
        raise ValueError(f"Unsupported train type: {train_type}")
    return trainers[train_type](config_manager, tune_config)

def tune_objective(config, config_manager, tune_config, train_type: str):
    """범용 튜닝 목표 함수"""
    try:
        # 1. 적절한 트레이너 선택
        trainer = get_tune_trainer(train_type, config_manager, tune_config)

        # 2. 데이터 준비
        train_dataset, eval_dataset = prepare_dataset(
            config_manager=config_manager,
            tokenizer=trainer.tokenizer,
            task_type=train_type
        )

        # 3. 학습 실행
        metrics = trainer.tune_train(train_dataset, eval_dataset, config)

        # 4. 결과 보고 (딕셔너리 형태로 수정)
        tune.report({
            "eval_loss": metrics['eval_loss'],
            "eval_accuracy": metrics['eval_accuracy']
        })

    except Exception as e:
        print(f"Error in tune_objective: {e}")
        # 실패한 트라이얼에 대해 높은 loss 보고 (딕셔너리 형태로 수정)
        tune.report({
            "eval_loss": float('inf'),
            "eval_accuracy": 0.0
        })
        raise e
