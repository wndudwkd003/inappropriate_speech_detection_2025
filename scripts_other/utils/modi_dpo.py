import json
import re
import os
from pathlib import Path

def extract_number_from_rejected(rejected_text):
    """
    rejected 텍스트에서 숫자를 추출합니다.
    예: "1. 손이 크다" -> "1"
    """
    if not rejected_text:
        return rejected_text

    # 정규식으로 숫자 추출 (맨 앞의 숫자만)
    match = re.match(r'^(\d+)', rejected_text.strip())
    if match:
        return match.group(1)

    return rejected_text

def process_json_file(file_path):
    """
    JSON 파일을 처리하여 rejected 필드의 숫자를 추출합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 각 항목의 rejected 필드 처리
    for item in data:
        if 'output' in item and 'rejected' in item['output']:
            original_rejected = item['output']['rejected']
            item['output']['rejected'] = extract_number_from_rejected(original_rejected)

    return data

def main():
    # 폴더 경로 설정
    folder_path = Path("data/other/kowiki_2025_results_for_dpo")

    # 파일 목록
    files_to_process = ["dev.json", "train.json"]

    for filename in files_to_process:
        file_path = folder_path / filename

        if not file_path.exists():
            print(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        print(f"처리 중: {filename}")

        # 백업 파일 생성
        backup_path = folder_path / f"{filename.split('.')[0]}_backup.json"

        # 원본 파일을 백업으로 복사
        with open(file_path, 'r', encoding='utf-8') as original:
            with open(backup_path, 'w', encoding='utf-8') as backup:
                backup.write(original.read())

        print(f"백업 생성: {backup_path}")

        # JSON 파일 처리
        processed_data = process_json_file(file_path)

        # 처리된 데이터를 원본 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"처리 완료: {filename}")
        print(f"총 {len(processed_data)}개 항목 처리됨\n")

if __name__ == "__main__":
    main()
