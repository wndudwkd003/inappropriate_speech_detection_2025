import json
import os
import shutil
import openai
import time
import yaml

DEBUG = False  # 디버깅 모드 설정

label_mapping = {
    "inappropriate": "부적절",
    "appropriate": "적절",
}

class CoTDataGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.input_dir = "data/other_for_conf"
        self.output_dir = "data/other_for_conf_for_cot_perspectives"

        # 4가지 추론 관점 정의
        self.perspectives = [
            "대화에서 사용된 언어 표현 중 부적절한 단어나 표현(욕설, 비속어)이 발견되었는지?",
            "대화가 부적절한 발화의 경우 어떤 원인(차별, 혐오, 편향, 폄하 등)에서 문제가 되는지?",
            "대화의 흐름을 고려했을 때 각 발화자의 발언이 대화 당사자와에게는 어떤 영향을주고, 제3자에게는 어떤 영향을 미치는지?",
            "대화의 전체 맥락과 발화 간의 상호작용을 고려하여 왜 각 발화가 적절 또는 부적절로 판단되었는지?",
        ]

    def setup_output_directory(self):
        """출력 디렉토리 설정 및 test.json 복사"""
        os.makedirs(self.output_dir, exist_ok=True)

        # test.json 그대로 복사
        test_src = os.path.join(self.input_dir, "test.json")
        test_dst = os.path.join(self.output_dir, "test.json")
        if os.path.exists(test_src):
            shutil.copy2(test_src, test_dst)
            print(f"test.json 복사 완료: {test_dst}")

    def generate_thinking_process_for_utterance(self, utterances, target_utterance_idx, label, perspective, max_retries=3):
        """특정 발화에 대한 특정 관점의 추론 과정 생성"""

        # 전체 대화 내용 포맷팅
        dialogue_text = ""
        for utterance in utterances:
            dialogue_text += f"{utterance['speaker_id']}: {utterance['form']}\n"

        # 타겟 발화 정보
        target_utterance = utterances[target_utterance_idx]
        target_speaker = target_utterance['speaker_id']
        target_text = target_utterance['form']

        prompt = f"""다음 대화에서 특정 발화에 대해 주어진 관점으로 추론해주세요.

[전체 대화 내용]
{dialogue_text}

[분석 대상 발화]
{target_speaker}: "{target_text}"

[이 발화의 정답]
{label}

[추론 관점]
{perspective}

요구사항:
1. 분석 대상 발화는 {label}입니다. 발화의 정답은 명확합니다. 주어진 추론 관점에 따라 지시하는 내용만 답하세요.
2. 2문장이내로 간결하고 핵심적인 추론 과정 내용만 작성하세요. 당신은 발화가 적절한지, 부적절한지 탐지하는 것이 아닙니다. 본인이 생각하기에 정답과 다르더라도 사족을 달지 마세요.
"""

        for attempt in range(max_retries):
            try:
                if DEBUG: print("prompt:", prompt)  # 디버깅용 출력
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": "당신은 대화 맥락에서 주어진 관점에 따라 정확하고 논리적인 분석을 제공하는 전문가 어시스턴트 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=512
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"GPT API 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  → {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"  → 최대 재시도 횟수 초과, 기본값 반환")
                    return f"{perspective}에 대한 분석을 수행했습니다."

    def load_existing_ids(self, filepath):
        """기존 저장된 데이터의 ID 목록 로드"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                return set(item['id'] for item in existing_data)
            except:
                return set()
        return set()

    def append_to_json_file(self, filepath, new_items):
        """JSON 파일에 새 항목들 추가"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(new_items)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def process_file(self, filename):
        """개별 파일 처리"""
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)

        if not os.path.exists(input_path):
            print(f"파일이 존재하지 않습니다: {input_path}")
            return

        existing_ids = self.load_existing_ids(output_path)
        print(f"기존 저장된 데이터: {len(existing_ids)}개")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_processed = 0
        skipped_count = 0

        for i, item in enumerate(data):
            print(f"\n{filename} 처리 중: {i+1}/{len(data)}")

            if item['id'] in existing_ids:
                skipped_count += 1
                print(f"  → 이미 처리됨: {item['id']} (스킵)")
                continue

            utterances = item["input"]["utterance"]
            labels = item["output"]

            # 각 발화에 대해 처리
            output_with_cot = []

            for j, (utterance, label_info) in enumerate(zip(utterances, labels)):
                print(f"  발화 {j+1}/{len(utterances)} 처리 중...")
                label_info["label_kr"] = label_mapping.get(label_info["label"], label_info["label"])

                # 4가지 관점에 대한 추론 생성
                cot_results = []
                for k, perspective in enumerate(self.perspectives):
                    print(f"    관점 {k+1}/4 추론 중...")
                    thinking = self.generate_thinking_process_for_utterance(
                        utterances, j, label_info["label_kr"], perspective
                    )
                    cot_results.append(thinking)
                    time.sleep(0.3)  # API 제한 방지

                # 출력 형식에 맞게 구성
                output_item = {
                    "id": label_info['id'],
                    "label": label_info['label'],
                    "label_kr": label_info['label_kr'],
                    "cot": cot_results  # 4가지 관점의 추론 결과 리스트
                }
                output_with_cot.append(output_item)

            # 원본 데이터 구조 유지하면서 output 업데이트
            cot_item = {
                "id": item['id'],
                "input": item["input"],
                "output": output_with_cot
            }

            # 즉시 파일에 저장
            self.append_to_json_file(output_path, [cot_item])
            total_processed += 1

            print(f"  → 저장 완료: {total_processed}개 데이터")

        print(f"\n{filename} 처리 완료: 전체 {len(data)}개 중 {total_processed}개 새로 처리, {skipped_count}개 스킵")

    def run(self):
        """전체 실행"""
        print("CoT 데이터 생성 시작... (4가지 관점)")
        print("관점:")
        for i, perspective in enumerate(self.perspectives, 1):
            print(f"  {i}. {perspective}")

        # 출력 디렉토리 설정
        self.setup_output_directory()

        # dev.json과 train.json 처리
        for filename in ["dev.json", "train.json"]:
            print(f"\n{'='*50}")
            print(f"{filename} 처리 시작...")
            print(f"{'='*50}")
            self.process_file(filename)

        print("\n모든 작업 완료!")

# 사용 예시
if __name__ == "__main__":
    # OpenAI API 키 설정
    with open("src/configs/tokens/token.yaml", 'r', encoding='utf-8') as f:
        tokens = yaml.safe_load(f)
    API_KEY = tokens["open_ai_token"]

    generator = CoTDataGenerator(API_KEY)
    generator.run()
