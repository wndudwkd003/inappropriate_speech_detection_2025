import unsloth
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from unsloth import FastLanguageModel
import os, json, argparse, hashlib, re
from src.configs.config_manager import ConfigManager
from src.data.prompt_manager import PromptManager
from src.data.base_dataset import make_chat
from src.utils.huggingface_utils import init_hub_env
from tqdm.auto import tqdm
from datetime import datetime

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

    model_path = cm.system.adapter_dir if cm.model.full_finetune else cm.model.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cm.model.max_seq_length,
        dtype=cm.model.dtype,
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True
    )

    # padding token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token

    # 어댑터 로드
    model = FastLanguageModel.for_inference(model)

    if not cm.model.full_finetune:
        model.load_adapter(cm.system.adapter_dir)

    # 테스트 데이터셋 로드
    with open(os.path.join(cm.system.data_raw_dir, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 시스템 프롬프트 가져오기
    prompt_version = cm.system.prompt_version
    system_prompt = PromptManager.get_system_prompt(prompt_version)

    # 결과를 저장할 리스트
    results = []

    for sample in tqdm(test_data, desc="Testing", unit="sample"):
        # make_chat 함수로 프롬프트 생성
        user_prompt = make_chat(
            sample["input"],
            cm
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 토크나이즈
        inputs = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)

        # 생성
        outputs = model.generate(
            inputs,
            max_new_tokens=cm.model.max_new_tokens,
            do_sample=cm.model.do_sample,
            attention_mask=attention_mask,
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
        else:
            parsed_output = {"answer": answer}

        # 결과 저장
        results.append({
            "id": sample["id"],
            "input": sample["input"],
            "output": parsed_output
        })

    # 결과 파일 저장
    save_dir_hash = hashlib.md5(cm.system.save_dir.encode()).hexdigest()[:8]  # 8자리만 사용
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 파일명에 해시 포함
    output_filename = f"test_results_{save_dir_hash}_{timestamp}.json"
    output_path = os.path.join(cm.system.test_result_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.dirname(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SFT Model")
    parser.add_argument("--save_dir", type=str, required=True, help="Must be set to save the trained model.")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager_for_test(save_dir=args.save_dir)
    config_manager.update_config("sft", {"seed": config_manager.system.seed})
    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    main(config_manager)
