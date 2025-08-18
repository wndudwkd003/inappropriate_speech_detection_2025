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
import torch, json
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

def init_config_manager_for_test(save_dir: str = "configs") -> ConfigManager:
    # 테스트 환경에서는 저장된 설정을 불러옴
    cm = ConfigManager()
    config_dir = os.path.join(save_dir, "configs")
    cm.load_all_configs(config_dir=config_dir)

    adapter_dir = os.path.join(save_dir, "lora_adapter")
    test_result_dir = os.path.join(save_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)
    print(f"Test results will be saved to: {test_result_dir}")

    cm.update_config("system", {
        "save_dir": save_dir,
        "adapter_dir": adapter_dir,
        "test_result_dir": test_result_dir
    })

    cm.print_all_configs()
    return cm


def main(cm: ConfigManager):
    # 테스트 모드

    add_args = dict()

    if cm.model.num_classes != -1:
        add_args["num_labels"] = cm.model.num_classes

    if not cm.system.only_decode:
        add_args["auto_model"] = AutoModelForSequenceClassification
        fast_model = FastModel
        print("Using FastModel for classification training.")
    else:
        fast_model = FastLanguageModel
        print("Using FastLanguageModel for generation training.")

    model_path = cm.system.adapter_dir if cm.model.full_finetune else cm.model.model_id
    model, tokenizer = fast_model.from_pretrained(
        model_name=model_path,
        max_seq_length=cm.model.max_seq_length,
        dtype=cm.model.dtype if cm.system.only_decode else None,
        load_in_4bit=cm.model.load_in_4bit,
        load_in_8bit=cm.model.load_in_8bit,
        full_finetuning=cm.model.full_finetune,
        trust_remote_code=True,
        **add_args
    )

    # padding token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token

    # 어댑터 로드
    model = fast_model.for_inference(model)
    # model.config.use_cache = False

    if not cm.model.full_finetune and cm.system.only_decode:
        model.load_adapter(cm.system.adapter_dir)

    original_data = {}
    test_file_path = os.path.join(cm.system.data_raw_dir, "test.json")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        original_test_data = json.load(f)
        # document_id를 키로 하는 딕셔너리 생성
        for item in original_test_data:
            original_data[item["id"]] = item["input"]

    # 테스트 데이터셋 로드
    test_dataset = prepare_dataset(
        config_manager=cm,
        tokenizer=tokenizer,
        task_type=CURRENT_TEST_TYPE,
        is_train=False
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # 결과를 document_id별로 그룹화할 딕셔너리
    document_results = defaultdict(lambda: {"id": "", "input": {}, "output": []})

    debug_count = 0
    for sample in tqdm(test_dataset, desc="Testing", unit="sample"):
        document_id = sample["document_id"]
        utterance_id = sample["utterance_id"]
        utterance_idx = sample["utterance_idx"]

        if cm.system.only_decode:
            # 생성형 모델 처리
            inputs = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

            # 생성
            outputs = model.generate(
                inputs,
                max_new_tokens=cm.model.max_new_tokens,
                do_sample=cm.model.do_sample,
                attention_mask=attention_mask,
                use_cache=cm.system.use_cache,
            )

            # 답변 추출
            answer = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

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
                parsed_output = {"think": ""}

            # 답변을 label로 변환
            predicted_label = convert_answer_to_label(final_answer, cm)

        else:
            # 분류 모델 처리
            inputs = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()

                # 디버깅: 처음 5개 샘플의 logits과 예측 결과 출력
                if debug_count < 5:
                    # BFloat16 에러 방지를 위해 float32로 변환
                    logits_float = logits.float()
                    softmax_probs = torch.softmax(logits_float, dim=-1)[0].cpu().numpy()
                    logits_values = logits_float[0].cpu().numpy()

                    print(f"\nSample {debug_count + 1}:")
                    print(f"  Document ID: {document_id}")
                    print(f"  Utterance ID: {utterance_id}")
                    print(f"  Logits: [{logits_values[0]:.4f}, {logits_values[1]:.4f}]")
                    print(f"  Predicted class: {predicted_class}")
                    print(f"  Softmax probs: [적절: {softmax_probs[0]:.4f}, 부적절: {softmax_probs[1]:.4f}]")
                    debug_count += 1

            predicted_label = "inappropriate" if predicted_class == 1 else "appropriate"

        # 결과를 document별로 저장
        if not document_results[document_id]["id"]:
            # 첫 번째 샘플에서 document 정보 초기화
            # 실제로는 원본 데이터에서 input 정보를 가져와야 함
            document_results[document_id]["id"] = document_id
            document_results[document_id]["input"] = original_data.get(document_id, {})

        # output 리스트에 결과 추가 (utterance_idx 순서대로)
        print(f"Document ID: {document_id}, Utterance ID: {utterance_id}, Predicted Label: {predicted_label}")
        output_item = {
            "id": utterance_id,
            "label": predicted_label,
            "utterance_idx": utterance_idx  # 정렬용
        }
        if cm.system.only_decode and cm.system.is_cot:
            output_item["think"] = parsed_output.get("think", "")

        document_results[document_id]["output"].append(output_item)

    # 결과를 최종 형태로 변환
    final_results = []
    for document_id, doc_data in document_results.items():
        # utterance_idx 순서대로 정렬
        doc_data["output"].sort(key=lambda x: x["utterance_idx"])

        # utterance_idx 제거 (최종 결과에는 불필요)
        for output in doc_data["output"]:
            del output["utterance_idx"]

        final_results.append({
            "id": doc_data["id"],
            "input": doc_data["input"],
            "output": doc_data["output"]
        })

    # 결과 파일 저장
    save_dir_hash = hashlib.md5(cm.system.save_dir.encode()).hexdigest()[:8]  # 8자리만 사용
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 파일명에 해시 포함
    output_filename = f"test_results_{save_dir_hash}_{timestamp}.json"
    output_path = os.path.join(cm.system.test_result_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.dirname(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SFT Model")
    parser.add_argument("--save_dir", type=str, required=True, help="Must be set to save the trained model.")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager_for_test(save_dir=args.save_dir)
    config_manager.update_config(CURRENT_TEST_TYPE, {"seed": config_manager.system.seed})
    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    # set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)
    main(config_manager)
