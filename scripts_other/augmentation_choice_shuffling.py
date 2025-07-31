import os
import json
import random
import shutil
from copy import deepcopy
from tqdm import tqdm
import re

def extract_choices_and_answer(question: str, answer: str):
    """
    숫자. 패턴으로 선택지를 추출하고 정답 위치를 찾아 반환
    """
    # 정규표현식으로 '1. xxx', '2. yyy' 등 추출
    pattern = r"(\d+)\.\s*([^0-9]+?)(?=(\d+\.)|$)"
    matches = list(re.finditer(pattern, question))

    if not matches:
        return None, None, None, None

    choices = [m.group(2).strip() for m in matches]
    choice_starts = [m.start() for m in matches]
    question_text = question[:choice_starts[0]].strip()

    answer_idx = int(answer) - 1
    if not (0 <= answer_idx < len(choices)):
        return None, None, None, None

    correct_text = choices[answer_idx]
    return question_text, choices, correct_text, answer_idx

def make_augmented_instance(orig_instance, aug_idx):
    q_full = orig_instance["input"]["question"]
    answer = orig_instance["output"]["answer"]

    q_text, choices, correct_text, answer_idx = extract_choices_and_answer(q_full, answer)
    if q_text is None:
        return None

    # 보기 셔플
    new_choices = deepcopy(choices)
    random.shuffle(new_choices)

    # 새로운 정답 번호
    try:
        new_answer = str(new_choices.index(correct_text) + 1)
    except ValueError:
        return None

    # 새로운 질문 문자열 조립
    new_choice_str = " ".join([f"{i+1}. {c}" for i, c in enumerate(new_choices)])
    new_question = f"{q_text} {new_choice_str}"

    # 새 인스턴스 구성
    new_instance = deepcopy(orig_instance)
    new_instance["id"] = f"{orig_instance['id']}_aug{aug_idx}"
    new_instance["input"]["question"] = new_question
    new_instance["output"]["answer"] = new_answer

    return new_instance

def main(
    src_dir="data/other/task_3_merged_1-3",
    dst_dir="data/other/task_3_merged_1-3_aug_s1",
    augment_times=2,
):
    os.makedirs(dst_dir, exist_ok=True)

    # dev, test는 그대로 복사
    for split in ["dev.json", "test.json"]:
        shutil.copyfile(os.path.join(src_dir, split), os.path.join(dst_dir, split))

    # train 데이터 로드
    with open(os.path.join(src_dir, "train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)

    new_data = []
    for inst in tqdm(train_data, desc="Augmenting train.json"):
        new_data.append(inst)

        if inst["input"].get("question_type", "") != "선다형":
            continue

        for aug_idx in range(1, augment_times + 1):
            aug_inst = make_augmented_instance(inst, aug_idx)
            if aug_inst:
                new_data.append(aug_inst)

    # 저장
    with open(os.path.join(dst_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main(augment_times=2)
