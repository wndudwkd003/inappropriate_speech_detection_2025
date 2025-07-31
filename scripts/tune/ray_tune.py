# scripts/tune/tune_sft.py
import argparse
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune import RunConfig
from src.configs.config_manager import ConfigManager, init_config_manager
from src.tune.tune_objective import tune_objective
from src.utils.huggingface_utils import init_hub_env
from src.utils.seed_utils import set_all_seeds
from transformers import set_seed
import os
import yaml

CURRENT_TRAIN_TYPE = "sft"

def main(config_manager: ConfigManager):
    tune_config = config_manager.tune

    # ray.init(
    #     num_gpus=1,
    #     num_cpus=4,
    #     ignore_reinit_error=True,
    #     runtime_env={
    #         "env_vars": {
    #             "CUDA_VISIBLE_DEVICES": "0",
    #             "CUDA_DEVICE_ORDER": "PCI_BUS_ID"
    #         }
    #     }
    # )

    ray.init(local_mode=True, ignore_reinit_error=True)

    # 4. 서치 스페이스 정의
    search_space = tune_config.to_search_space()
    # 5. 스케줄러 설정
    scheduler = ASHAScheduler(**tune_config.asha_config)
    # 6. 서치 알고리즘 설정 (수정)

    search_alg = OptunaSearch(
    metric=tune_config.asha_config["metric"],  # "eval_loss"
    mode=tune_config.asha_config["mode"]       # "min"
)

    # 7. Ray Tune 실행
    tuner = tune.Tuner(
        tune.with_parameters(
            tune_objective,
            config_manager=config_manager,
            tune_config=tune_config,
            train_type=CURRENT_TRAIN_TYPE
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config.num_samples,
            max_concurrent_trials=tune_config.max_concurrent_trials,
            scheduler=scheduler,
            search_alg=search_alg
        ),
        run_config=RunConfig(
            name=tune_config.name,
            storage_path=tune_config.local_dir
        )
    )


    results = tuner.fit()

    # 8. 결과 분석 및 저장
    try:
        best_result = results.get_best_result("eval_loss", "min")
        print(f"Best trial config: {best_result.config}")
        print(f"Best trial final validation loss: {best_result.metrics['eval_loss']}")

        # 9. 최적 하이퍼파라미터 저장
        save_best_hyperparameters(best_result, tune_config)

    except RuntimeError as e:
        print(f"모든 트라이얼이 실패했습니다: {e}")

        # 실패한 트라이얼 정보 출력
        failed_trials = [trial for trial in results if trial.error]
        print(f"실패한 트라이얼 수: {len(failed_trials)}")

        if failed_trials:
            print("첫 번째 실패 트라이얼의 에러:")
            print(failed_trials[0].error)

        # Ray 정리
        ray.shutdown()

def save_best_hyperparameters(best_result, tune_config):
    """최적 하이퍼파라미터를 YAML 파일로 저장"""
    # 최적 설정 정리
    best_params = {
        'best_hyperparameters': best_result.config,
        'best_metrics': {
            'eval_loss': best_result.metrics.get('eval_loss'),
            'eval_accuracy': best_result.metrics.get('eval_accuracy', 0)
        }
    }

    # 저장 경로
    save_path = os.path.join(tune_config.output_dir, "best_hyperparameters.yaml")

    # YAML 파일로 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_params, f, allow_unicode=True, default_flow_style=False)

    print(f"Best hyperparameters saved to: {save_path}")

def update_config_for_tune(config_manager: ConfigManager):
    # 절대 경로로 변환
    data_raw_dir = config_manager.system.data_raw_dir
    data_raw_dir = os.path.abspath(data_raw_dir)

    if CURRENT_TRAIN_TYPE == "sft":
        output_dir = config_manager.sft.output_dir
    elif CURRENT_TRAIN_TYPE == "dpo":
        output_dir = config_manager.dpo.output_dir

    output_time_dir = os.path.basename(output_dir)
    output_dir = os.path.join(config_manager.tune.output_dir, output_time_dir)
    local_dir = os.path.join(config_manager.tune.local_dir, output_time_dir)

    # make dir
    output_dir = os.path.abspath(output_dir)
    local_dir = os.path.abspath(local_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    config_manager.update_config("tune", {
        "output_dir": output_dir,
        "local_dir": local_dir,
    })
    config_manager.update_config("system", {
        "data_raw_dir": data_raw_dir,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Model Hyperparameters")
    parser.add_argument("--config", type=str, default="configs", help="Path to configuration directory")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager(dir=args.config, train_type=CURRENT_TRAIN_TYPE, is_tune=True)
    config_manager.update_config(CURRENT_TRAIN_TYPE, {"seed": config_manager.system.seed})
    update_config_for_tune(config_manager)

    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    # 메인 함수 실행
    main(config_manager)
