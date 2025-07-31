import openai
import os
import json
import yaml
import shutil
from typing import List
from tqdm import tqdm
import time
import re

class OpenAILlmAugmentation:
    def __init__(
        self,
        token_path: str = "src/configs/tokens/token.yaml",
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        OpenAI GPT를 사용한 질문 증강 클래스

        Args:
            token_path (str): OpenAI API 토큰이 저장된 yaml 파일 경로
            model (str): 사용할 GPT 모델
            temperature (float): 생성 온도
            max_tokens (int): 최대 토큰 수
        """
        self.api_key = self.load_openai_token(token_path)
        openai.api_key = self.api_key

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load_openai_token(self, token_path: str) -> str:
        """YAML 파일에서 OpenAI 토큰 로드"""
        with open(token_path, "r") as f:
            token_data = yaml.safe_load(f)
        return token_data["open_ai_token"]

    def generate_paraphrased_questions(self, questions: List[str]) -> List[str]:
        """
        질문 배열을 받아, 각 질문별로 의미는 동일하되 표현은 다른 질문으로 변형

        Args:
            questions (List[str]): 변형할 질문들

        Returns:
            List[str]: 변형된 질문들
        """

        system_prompt = """
        당신은 뛰어난 언어 능력을 가진 문장가입니다.
        주어진 질문들을 의미는 동일하되 표현만 다르게 변형해주세요.
        """

        user_prompt = f"""
        다음의 주어진 문장들을 의미는 동일하되 표현만 다르게 변형해주세요.
        결과는 JSON 형식의 문자열 리스트로만 출력해주세요.
        주의: 숫자와 고유명사(지명, 기관명, 인명 등)는 절대 바꾸지 마세요.

        아래의 예시는 참고용입니다.

        입력:
        [
            "오늘은 비가 많이 내린다.",
            "즐거운 추석 명절 보내시기 바랍니다.",
            "나랑 사귀자!",
            "피곤할 땐 아메리카노를 마시면 좋지.",
            "국토교통부가 디딤돌 대출 규제를 잠정 유예한다."
        ]

        출력:
        [
            "금일은 강수량이 많다."
            "한가위 명절 재미있게 보내세요."
            "우리 오늘부터 1일이야~"
            "졸리면 커피 한잔해~",
            "국토교통부가 디딤돌 대출 규제를 잠시 미루기로 결정하였다."
        ]

        이 예시를 바탕으로 다음의 주어진 문장들의 표현을 다르게 변형해주세요.

        {json.dumps(questions, ensure_ascii=False)}
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = response.choices[0].message.content.strip()

            # JSON 추출
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                paraphrased_questions = json.loads(json_text)

                # 길이 확인
                if len(paraphrased_questions) == len(questions):
                    return paraphrased_questions
                else:
                    print(f"Warning: Length mismatch. Expected {len(questions)}, got {len(paraphrased_questions)}")
                    return questions
            else:
                print("Warning: Could not extract JSON from response")
                return questions

        except Exception as e:
            print(f"Error in API call: {e}")
            return questions

def load_sft_data(file_path: str) -> List[dict]:
    """SFT 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_sft_data(data: List[dict], file_path: str):
    """SFT 데이터 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_question_types(data: List[dict], target_types: List[str] = ["단답형", "서술형"]) -> List[dict]:
    """단답형과 서술형만 필터링"""
    filtered_data = []
    for item in data:
        if item["input"]["question_type"] in target_types:
            filtered_data.append(item)
    return filtered_data

def apply_llm_question_augmentation(
    data: List[dict],
    augmentor: OpenAILlmAugmentation,
    output_file: str,
    batch_size: int = 5,
    delay: float = 1.0
) -> List[dict]:
    """
    LLM을 사용한 question 증강 - 배치별 즉시 저장

    Args:
        data: 증강할 데이터
        augmentor: OpenAI 증강기
        output_file: 출력 파일 경로
        batch_size: 배치 크기
        delay: API 호출 간 지연시간 (초)
    """
    # 기존 파일이 있다면 로드, 없다면 원본 데이터로 시작
    if os.path.exists(output_file):
        print(f"Loading existing file: {output_file}")
        current_data = load_sft_data(output_file)

        # 이미 증강된 아이템들의 원본 ID 찾기
        augmented_ids = set()
        for item in current_data:
            if item["id"].endswith("_llm_aug"):
                original_id = item["id"].replace("_llm_aug", "")
                augmented_ids.add(original_id)

        # 아직 증강되지 않은 데이터만 필터링
        remaining_data = [item for item in data if item["id"] not in augmented_ids]
        print(f"Found {len(augmented_ids)} already processed items")
        print(f"Remaining items to process: {len(remaining_data)}")

        if len(remaining_data) == 0:
            print("All items already processed!")
            return current_data
    else:
        # 새로운 파일: 원본 데이터로 시작
        print("Creating new augmented file...")
        # 원본 데이터에서 증강 대상이 아닌 것들 먼저 저장
        original_train_data = load_sft_data('/workspace/kr_llm_ft_25/data/other/task_3_merged_1-3/train.json')
        current_data = original_train_data.copy()
        save_sft_data(current_data, output_file)
        remaining_data = data
        print(f"Initialized with {len(current_data)} original items")

    print(f"Processing {len(remaining_data)} items in batches of {batch_size}")

    # 배치 단위로 처리하고 즉시 저장
    for i in tqdm(range(0, len(remaining_data), batch_size), desc="LLM augmentation"):
        batch = remaining_data[i:i + batch_size]
        questions = [item["input"]["question"] for item in batch]

        try:
            # 배치 단위로 question 증강
            augmented_questions = augmentor.generate_paraphrased_questions(questions)

            # 배치 결과를 current_data에 추가
            batch_augmented = []
            for j, item in enumerate(batch):
                new_item = item.copy()
                new_item["input"] = item["input"].copy()
                new_item["output"] = item["output"].copy()

                # 증강된 question 적용
                new_item["input"]["question"] = augmented_questions[j]
                new_item["id"] = f"{item['id']}_llm_aug"

                batch_augmented.append(new_item)

            # current_data에 추가하고 즉시 저장
            current_data.extend(batch_augmented)
            save_sft_data(current_data, output_file)

            print(f"Batch {i//batch_size + 1} completed and saved. Total items: {len(current_data)}")

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # 에러 발생시 원본 데이터로 추가하고 저장
            batch_fallback = []
            for item in batch:
                new_item = item.copy()
                new_item["id"] = f"{item['id']}_llm_aug"
                batch_fallback.append(new_item)

            current_data.extend(batch_fallback)
            save_sft_data(current_data, output_file)
            print(f"Batch {i//batch_size + 1} failed but saved with original questions")

        # API 호출 제한 방지를 위한 지연
        if i + batch_size < len(remaining_data):
            time.sleep(delay)

    return current_data

def main():
    # 경로 설정
    input_dir = '/workspace/kr_llm_ft_25/data/other/task_3_merged_1-3'
    output_dir = '/workspace/kr_llm_ft_25/data/other/task_3_merged_1-3_s3'
    token_path = "src/configs/tokens/token.yaml"

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # dev, test 파일 복사
    for file_name in ['dev.json', 'test.json']:
        src_path = os.path.join(input_dir, file_name)
        dst_path = os.path.join(output_dir, file_name)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file_name}")

    # train 데이터 로드
    train_path = os.path.join(input_dir, 'train.json')
    print("Loading train data...")
    train_data = load_sft_data(train_path)
    print(f"Loaded {len(train_data)} items")

    # 단답형과 서술형만 필터링
    print("Filtering question types...")
    filtered_data = filter_question_types(train_data, ["단답형", "서술형"])
    print(f"Filtered data: {len(filtered_data)} items (단답형, 서술형 only)")

    # 출력 파일 경로
    output_file = os.path.join(output_dir, 'train.json')  # 원래 저장하려던 곳에 바로 저장

    # OpenAI LLM 증강기 초기화
    print("Initializing OpenAI augmentor...")
    try:
        augmentor = OpenAILlmAugmentation(
            token_path=token_path,
            model="gpt-4.1-2025-04-14",  # 또는 "gpt-3.5-turbo"
            temperature=0.7
        )
        print("OpenAI augmentor initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI augmentor: {e}")
        return

    # 사용자 확인
    if os.path.exists(output_file):
        print(f"\nExisting file found: {output_file}")
        print("Will continue from where it left off...")
    else:
        print(f"\nReady to augment {len(filtered_data)} questions using OpenAI GPT")
        print("Warning: This will incur API costs!")

        user_input = input("Continue with LLM augmentation? (y/n): ")
        if user_input.lower() != 'y':
            print("Augmentation cancelled")
            return

    # LLM 증강 실행 (배치별 즉시 저장)
    print("\nStarting LLM augmentation...")
    final_data = apply_llm_question_augmentation(
        filtered_data,
        augmentor,
        output_file,
        batch_size=5,  # 배치 크기 (API 제한에 따라 조정)
        delay=1.0      # API 호출 간 지연시간
    )

    print(f"\nAugmentation completed!")
    print(f"Final data saved to: {output_file}")
    print(f"Total items in final file: {len(final_data)}")

if __name__ == "__main__":
    main()
