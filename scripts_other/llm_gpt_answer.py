import unsloth
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from unsloth import FastLanguageModel, FastModel
import os, json, argparse, hashlib, re
from src.configs.config_manager import ConfigManager
from src.data.prompt_manager import PromptManager
from src.utils.data_utils import prepare_dataset
from src.utils.huggingface_utils import init_hub_env
from tqdm.auto import tqdm
from datetime import datetime
from collections import defaultdict
import torch
from datetime import datetime
import yaml
import openai
from transformers import AutoModelForSequenceClassification


CURRENT_TEST_TYPE = "sft"


def parse_cot_answer(answer: str) -> dict:
    """CoT 답변을 파싱하여 think와 answer 부분을 분리"""
    result = {}

    # think 태그 추출
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, answer, re.DOTALL)

    if think_match:
        think_content = think_match.group(1)
        # 양옆 \n 제거 및 중간 \n을 공백으로 변경
        think_content = think_content.strip()
        think_content = re.sub(r'\n+', ' ', think_content)
        result["think"] = think_content
    else:
        result["think"] = ""

    # answer 태그 추출
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, answer, re.DOTALL)

    if answer_match:
        answer_content = answer_match.group(1)
        # 양옆 \n 제거 및 중간 \n을 공백으로 변경
        answer_content = answer_content.strip()
        answer_content = re.sub(r'\n+', ' ', answer_content)
        result["answer"] = answer_content
    else:
        # answer 태그가 없으면 전체 텍스트에서 think 태그 부분만 제거
        clean_answer = re.sub(think_pattern, '', answer, flags=re.DOTALL).strip()
        clean_answer = re.sub(r'\n+', ' ', clean_answer)
        result["answer"] = clean_answer

    return result

def convert_answer_to_label(answer: str, config_manager: ConfigManager) -> str:
    """모델 답변을 적절한 label로 변환"""
    answer = answer.strip().lower()

    if config_manager.system.only_decode:
        # 생성형 모델의 경우
        if "부적절" in answer or "inappropriate" in answer:
            return "inappropriate"
        elif "적절" in answer or "appropriate" in answer:
            return "appropriate"
        else:
            # 기본값 또는 예외 처리
            print(f"Warning: Could not parse answer '{answer}', defaulting to 'appropriate'")
            return "appropriate"
    else:
        # 분류 모델의 경우 - 이 부분은 별도 처리 필요
        # logits에서 예측값을 받아와야 함
        return answer

def init_config_manager_for_llm() -> ConfigManager:
    # 테스트 환경에서는 저장된 설정을 불러옴
    cm = ConfigManager()
    config_dir = os.path.join("configs")
    cm.load_all_configs(config_dir=config_dir)

    save_dir = os.path.join("output_llm", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    test_result_dir = os.path.join(save_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)
    print(f"Test results will be saved to: {test_result_dir}")

    with open('src/configs/tokens/token.yaml', 'r') as f:
        token_config = yaml.safe_load(f)
        openai.api_key = token_config.get("open_ai_token", "")

    cm.update_config("system", {
        "save_dir": save_dir,
        "test_result_dir": test_result_dir
    })

    cm.print_all_configs()
    return cm


def main(cm: ConfigManager):
    # 테스트 모드

    # 테스트 데이터셋 로드
    test_dataset = prepare_dataset(
        config_manager=cm,
        tokenizer=None,
        task_type=CURRENT_TEST_TYPE,
        is_train=False
    )

    # 결과를 document_id별로 그룹화할 딕셔너리
    document_results = defaultdict(lambda: {"id": "", "input": {}, "output": []})

    # ----------------------[추가] 매 처리 시 저장할 출력 경로 미리 고정 ----------------------
    save_dir_hash = hashlib.md5(cm.system.save_dir.encode()).hexdigest()[:8]  # 8자리만 사용
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"test_results_{save_dir_hash}_{timestamp}.json"
    output_path = os.path.join(cm.system.test_result_dir, output_filename)
    # -------------------------------------------------------------------------------------

    # ----------------------[추가] 문서 결과를 최종 포맷으로 만드는 보조 함수 -------------------
    def build_final_results(_doc_results: dict):
        final_results_snapshot = []
        for _document_id, _doc_data in _doc_results.items():
            # 정렬만 하고, 원본을 변형하지 않도록 새로운 리스트 생성
            _sorted = sorted(_doc_data["output"], key=lambda x: x["utterance_idx"])
            _outputs = [{"id": o["id"], "label": o["label"]} for o in _sorted]
            final_results_snapshot.append({
                "id": _doc_data["id"],
                "input": _doc_data["input"],
                "output": _outputs
            })
        return final_results_snapshot
    # -------------------------------------------------------------------------------------

    debug_count = 0
    for sample in tqdm(test_dataset, desc="Testing", unit="sample"):
        system_prompt = sample["system_prompt"]
        user_content = sample["user_content"]
        document_id = sample["document_id"]
        utterance_id = sample["utterance_id"]
        utterance_idx = sample["utterance_idx"]

        response = openai.ChatCompletion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=cm.model.max_new_tokens
        )

        answer = response.choices[0].message.content.strip()

        if answer.startswith("답변: "):
            answer = answer[4:]
        elif answer.startswith("답변:"):
            answer = answer[3:]

        if "#" in answer:
            answer = answer.split("#")[0].strip()

        # CoT 파싱 (is_cot가 True인 경우)
        if cm.system.is_cot:
            parsed_output = parse_cot_answer(answer)
            final_answer = parsed_output["answer"]
        else:
            final_answer = answer

        # 답변을 label로 변환
        predicted_label = convert_answer_to_label(final_answer, cm)

        # 결과를 document별로 저장
        if not document_results[document_id]["id"]:
            # 첫 번째 샘플에서 document 정보 초기화
            # 실제로는 원본 데이터에서 input 정보를 가져와야 함
            document_results[document_id]["id"] = document_id
            # document_results[document_id]["input"] = sample["original_input"]  # 이 부분은 원본 input 정보 필요

        # output 리스트에 결과 추가 (utterance_idx 순서대로)
        print(f"Document ID: {document_id}, Utterance ID: {utterance_id}, Predicted Label: {predicted_label}")
        document_results[document_id]["output"].append({
            "id": utterance_id,
            "label": predicted_label,
            "utterance_idx": utterance_idx  # 정렬용
        })

        # ----------------------[추가] 각 샘플 처리 직후, 현재까지 결과 즉시 저장 ----------------------
        final_results_now = build_final_results(document_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_now, f, ensure_ascii=False, indent=2)
        # -------------------------------------------------------------------------------------------

    # 결과를 최종 형태로 변환 (최종에도 한 번 더 저장)
    final_results = build_final_results(document_results)

    # 결과 파일 저장 (동일 파일에 다시 기록)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.dirname(output_path)}")


if __name__ == "__main__":
    # 설정 관리자 초기화
    config_manager = init_config_manager_for_llm()
    config_manager.update_config(CURRENT_TEST_TYPE, {"seed": config_manager.system.seed})
    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    # set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)
    main(config_manager)
