import os
import json
import shutil
from typing import List, Dict, Tuple
from collections import defaultdict

def load_sft_data(file_path: str) -> List[dict]:
    """SFT 데이터 로드"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_sft_data(data: List[dict], file_path: str):
    """SFT 데이터 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_content_hash(item: dict) -> str:
    """
    아이템의 내용을 기반으로 해시 생성 (question, answer, question_type 등)
    ID는 제외하고 실제 내용만으로 해시 생성
    """
    # 내용 추출
    input_data = item.get("input", {})
    output_data = item.get("output", {})

    # 중요한 내용들만 추출해서 해시 생성
    content_parts = [
        input_data.get("question", ""),
        input_data.get("question_type", ""),
        input_data.get("context", ""),
        output_data.get("answer", ""),
        str(output_data.get("confidence", "")),
    ]

    # 빈 문자열 제거하고 정규화
    content_parts = [str(part).strip() for part in content_parts if part]
    content_string = "|".join(content_parts)

    return hash(content_string)

def merge_train_datasets(source_dirs: List[str], target_dir: str):
    """
    여러 소스 디렉토리의 train.json을 병합

    Args:
        source_dirs: 소스 디렉토리 목록 (s1, s2, s3 등)
        target_dir: 타겟 디렉토리 (c1)
    """

    # 타겟 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)

    # 모든 train 데이터 로드
    all_train_data = []

    for source_dir in source_dirs:
        train_file = os.path.join(source_dir, 'train.json')
        if os.path.exists(train_file):
            data = load_sft_data(train_file)
            print(f"Loaded {len(data)} items from {source_dir}")
            all_train_data.extend(data)
        else:
            print(f"Warning: {train_file} not found")

    print(f"Total items before deduplication: {len(all_train_data)}")

    # 중복 제거 로직
    # 내용 해시 -> 아이템 리스트 매핑
    content_to_items = defaultdict(list)

    for item in all_train_data:
        content_hash = create_content_hash(item)
        content_to_items[content_hash].append(item)

    # 중복 제거된 최종 데이터
    merged_data = []
    duplicate_count = 0

    for content_hash, items in content_to_items.items():
        if len(items) == 1:
            # 유일한 아이템
            merged_data.append(items[0])
        else:
            # 내용이 같은 여러 아이템이 있음 - 하나만 선택
            # 원본 데이터를 우선시 (id에 _aug가 없는 것)
            original_items = [item for item in items if not item["id"].endswith("_llm_aug")]

            if original_items:
                # 원본이 있으면 원본 중 첫 번째 선택
                merged_data.append(original_items[0])
            else:
                # 모두 증강 데이터면 첫 번째 선택
                merged_data.append(items[0])

            duplicate_count += len(items) - 1

            # 중복 정보 출력 (처음 몇 개만)
            if len(merged_data) <= 5:
                print(f"Found {len(items)} duplicate items with content hash {content_hash}")
                for item in items:
                    print(f"  - ID: {item['id']}")

    print(f"Removed {duplicate_count} duplicate items")
    print(f"Final merged dataset: {len(merged_data)} items")

    # 저장
    output_file = os.path.join(target_dir, 'train.json')
    save_sft_data(merged_data, output_file)
    print(f"Merged train data saved to: {output_file}")

    return merged_data

def copy_dev_test_files(source_dirs: List[str], target_dir: str):
    """
    dev.json과 test.json 파일들을 복사 (첫 번째로 찾은 것 사용)
    """

    # 타겟 디렉토리 생성 (이 부분이 누락되어 있었음)
    os.makedirs(target_dir, exist_ok=True)

    for file_name in ['dev.json', 'test.json']:
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(target_path):
            print(f"{file_name} already exists in target directory")
            continue

        # 첫 번째로 찾은 파일 복사
        copied = False
        for source_dir in source_dirs:
            source_path = os.path.join(source_dir, file_name)
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copied {file_name} from {source_dir}")
                copied = True
                break

        if not copied:
            print(f"Warning: {file_name} not found in any source directory")

def analyze_dataset_composition(data: List[dict]):
    """데이터셋 구성 분석"""

    print("\n=== Dataset Composition Analysis ===")

    # ID 패턴 분석
    original_count = 0
    llm_aug_count = 0
    other_aug_count = 0

    question_types = defaultdict(int)

    for item in data:
        item_id = item["id"]
        if item_id.endswith("_llm_aug"):
            llm_aug_count += 1
        elif "_aug" in item_id or "_paraphrase" in item_id:
            other_aug_count += 1
        else:
            original_count += 1

        # 질문 유형 분석
        question_type = item.get("input", {}).get("question_type", "unknown")
        question_types[question_type] += 1

    print(f"Original data: {original_count}")
    print(f"LLM augmented data: {llm_aug_count}")
    print(f"Other augmented data: {other_aug_count}")
    print(f"Total: {len(data)}")

    print("\nQuestion types distribution:")
    for qtype, count in sorted(question_types.items()):
        print(f"  {qtype}: {count}")

def main():
    # 경로 설정
    base_dir = '/workspace/kr_llm_ft_25/data/other'
    source_dirs = [
        os.path.join(base_dir, 'task_3_merged_1-3_s1'),
        os.path.join(base_dir, 'task_3_merged_1-3_s2'),
        os.path.join(base_dir, 'task_3_merged_1-3_s3')
    ]
    target_dir = os.path.join(base_dir, 'task_3_merged_1-3_c1')

    print("=== Dataset Merging Script ===")
    print(f"Source directories: {source_dirs}")
    print(f"Target directory: {target_dir}")

    # 소스 디렉토리 존재 확인
    existing_dirs = []
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            existing_dirs.append(source_dir)
            print(f"✓ Found: {source_dir}")
        else:
            print(f"✗ Missing: {source_dir}")

    if not existing_dirs:
        print("Error: No source directories found!")
        return

    print(f"\nWill merge {len(existing_dirs)} directories")

    # 사용자 확인
    user_input = input("\nProceed with merging? (y/n): ")
    if user_input.lower() != 'y':
        print("Merging cancelled")
        return

    # dev, test 파일 복사
    print("\n=== Copying dev and test files ===")
    copy_dev_test_files(existing_dirs, target_dir)

    # train 데이터 병합
    print("\n=== Merging train datasets ===")
    merged_data = merge_train_datasets(existing_dirs, target_dir)

    # 데이터셋 구성 분석
    analyze_dataset_composition(merged_data)

    print(f"\n=== Merging completed! ===")
    print(f"Results saved to: {target_dir}")

if __name__ == "__main__":
    main()
