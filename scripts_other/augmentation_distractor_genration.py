import os
import json
import yaml
import openai
import random
import shutil
from tqdm import tqdm
from copy import deepcopy
import re

def load_openai_token(token_path="src/configs/tokens/token.yaml"):
    with open(token_path, "r") as f:
        token_data = yaml.safe_load(f)
    return token_data["open_ai_token"]

def extract_question_elements(question: str, answer_idx: int):
    pattern = r"(\d+)\.\s*([^0-9]+?)(?=(\d+\.)|$)"
    matches = list(re.finditer(pattern, question))
    if not matches or answer_idx >= len(matches):
        return None, None, None
    question_text = question[:matches[0].start()].strip()
    choices = [m.group(2).strip() for m in matches]
    correct_text = choices[answer_idx]
    return question_text, correct_text, choices

def generate_distractors_fixed_count(question_text, correct_text, num_distractors):
    prompt = f"""당신은 한국 문화에 대한 객관식 문제를 만드는 AI입니다.

질문: {question_text}
정답: {correct_text}

정답과 유사하지만 틀린 오답 보기를 {num_distractors}개 만들어주세요.
오답은 간결하고 정답처럼 보이지만 정확히는 틀려야 합니다.
형식: JSON 배열 ["오답1", "오답2", ...]
"""
    response = openai.ChatCompletion.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    try:
        text = response.choices[0].message["content"]
        distractors = json.loads(text)
        return distractors
    except Exception as e:
        print(f"[오류] Distractor 파싱 실패: {e}")
        return None

def make_augmented_instance(instance, aug_idx, new_distractors, correct_text):
    q_text = instance["input"]["question"].split("1.")[0].strip()
    all_choices = new_distractors + [correct_text]
    random.shuffle(all_choices)
    new_question = q_text + " " + " ".join(f"{i+1}. {c}" for i, c in enumerate(all_choices))
    new_answer = str(all_choices.index(correct_text) + 1)

    new_instance = deepcopy(instance)
    new_instance["id"] = f"{instance['id']}_aug{aug_idx}"
    new_instance["input"]["question"] = new_question
    new_instance["output"]["answer"] = new_answer
    return new_instance

def read_existing_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            original_ids = {inst["id"].split("_aug")[0] for inst in data if "_aug" in inst["id"]}
            return original_ids
        except Exception:
            return set()

def append_instance(path, instance):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([instance], f, ensure_ascii=False, indent=2)
    else:
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(instance)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)

def main(
    src_dir="data/other/task_3_merged_1-3",
    dst_dir="data/other/task_3_merged_1-3_s2",
    token_path="src/configs/tokens/token.yaml",
    augment_times=1,
    max_augments=None
):
    os.makedirs(dst_dir, exist_ok=True)
    openai.api_key = load_openai_token(token_path)

    # dev/test 복사
    for fname in ["dev.json", "test.json"]:
        shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    # 기존 증강된 ID 로딩
    train_dst_path = os.path.join(dst_dir, "train.json")
    augmented_ids = read_existing_ids(train_dst_path)

    # train 데이터 로딩
    with open(os.path.join(src_dir, "train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)

    error_count = 0
    for idx, inst in enumerate(tqdm(train_data, desc="Generating distractors")):
        if inst["id"] in augmented_ids:
            continue

        if inst["input"].get("question_type", "") != "선다형":
            append_instance(train_dst_path, inst)
            continue

        answer_idx = int(inst["output"]["answer"]) - 1
        q_text, correct_text, all_choices = extract_question_elements(inst["input"]["question"], answer_idx)
        if not correct_text or not all_choices or len(all_choices) <= 1:
            append_instance(train_dst_path, inst)
            continue

        append_instance(train_dst_path, inst)

        for i in range(augment_times):
            num_distractors = len(all_choices) - 1
            distractors = generate_distractors_fixed_count(q_text, correct_text, num_distractors)
            print(f"[{idx}] 생성된 오답: {distractors}")
            if distractors:
                aug_inst = make_augmented_instance(inst, i + 1, distractors, correct_text)
                if aug_inst:
                    append_instance(train_dst_path, aug_inst)
            else:
                error_count += 1

    print(f"총 오류 수: {error_count}")

if __name__ == "__main__":
    main(augment_times=1)
