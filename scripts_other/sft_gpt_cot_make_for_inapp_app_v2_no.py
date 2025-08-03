import json
import os
import shutil
import openai
import time
import yaml
import re

DEBUG = True  # 디버깅 모드 설정

SPACE_RUN = re.compile(r'[ \t\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]{2,}')

def collapse_spaces(s: str) -> str:
    return SPACE_RUN.sub(' ', s)

label_mapping = {
    "inappropriate": "부적절",
    "appropriate": "적절",
}

class CoTDataGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.input_dir = "data/other_for_conf"
        self.output_dir = "data/other_for_conf_for_cot_perspectives_v2_no"  # 그대로 사용

    def setup_output_directory(self):
        """출력 디렉토리 설정 및 test.json 복사"""
        os.makedirs(self.output_dir, exist_ok=True)

        # test.json 그대로 복사
        test_src = os.path.join(self.input_dir, "test.json")
        test_dst = os.path.join(self.output_dir, "test.json")
        if os.path.exists(test_src):
            shutil.copy2(test_src, test_dst)
            print(f"test.json 복사 완료: {test_dst}")

    # 변경: 1개 → 4개 생성 후 리스트로 반환
    def generate_thinking_process_for_utterance(self, utterances, target_utterance_idx, label, max_retries=3):
        """특정 발화에 대해 정답(label)에 도달하기 위한 중간 추론 과정 4개 생성 (관점 없음)"""

        # 전체 대화 내용 포맷팅
        dialogue_text = ""
        for idx, utterance in enumerate(utterances):
            dialogue_text += f"{utterance['speaker_id']}_{idx+1}: {utterance['form']}\n"

        # 타겟 발화 정보
        target_utterance = utterances[target_utterance_idx]
        target_speaker = target_utterance['speaker_id']
        target_text = target_utterance['form']

        prompt = f"""
[부적절 발언 배경]
‘부적절 발언’이란 화자가 특정 대상에 대하여 공격성, 비하성, 차별성, 편향성을 표출하는 발언으로,
문서 내의 비속어, 욕설, 혹은 특정 대상에 대해 언어 표현으로 명백하게 드러난 발언뿐만 아니라 언어 표현 자체가 부적절하지 않더라도
맥락에 따라 화자의 의도가 대상에 대하여 공격성, 비하성, 차별성, 편향성을 표출하는 발언을 두루 포함합니다.

[발화 예시]
P1_1: 도로가 막혀서 늦은 걸 가지고 버스 기사한테 욕설을 퍼붓는 동네라면 운행을 중단해도 할 말 없지. => 적절,
P2_2: 버스기사는 싹 다 복날에 개 패듯 처맞아야 되는데? ㅋ => 부적절,
P1_2_3: 직설적으로 그러면 실례에요. => 적절
P2_4: 버스기사는 괴롭힘 당할 만 한데? ㅋ => 부적절

[중요 사항]
다음 대화에서 특정 발화에 대해, 주어진 정답에 도달하기 위한 '중간 추론 과정'만 간결히 작성하세요.

[전체 대화 내용]
{dialogue_text}

[분석 대상 발화]
{target_speaker}: "{target_text}"

[이 발화의 정답]
{label}

[필수요구사항]
1. 분석 대상 발화는 {label}입니다. 발화의 정답은 명확합니다.
2. 2문장 내외로 간결하고 핵심적인 추론 과정 내용만 작성하세요. 당신은 발화가 적절한지, 부적절한지 탐지하는 것이 아닙니다.
3. 발화 예시를 참고하지만 답변에 언급하지 마세요. 중요 사항 이후에 등장하는 대화 내용과 분석 대상 발화에 대해서만 답변하세요.
4. 본인이 생각하기에 정답과 다르더라도 사족을 달지 마세요.
5. 전체 대화의 맥락을 고려해서 정답이 도출되기 위한 사고 과정을 단계적으로 생각하세요.
"""

        for attempt in range(max_retries):
            try:
                if DEBUG:
                    print("prompt:", prompt)  # 디버깅용 출력

                # n=4로 4개 결과 한 번에 생성
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": "당신은 대화 맥락에서 주어진 정답에 도달하기 위한 중간 추론을 간결하고 논리적으로 작성하는 전문가 어시스턴트입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    n=4
                )

                # 4개 응답을 정규화하여 리스트로 반환
                outs = []
                for ch in response.choices:
                    t = (ch.message.content or "").replace("\n", " ").strip()
                    t = collapse_spaces(t)
                    outs.append(t)

                # 혹시 4개 미만이 돌아오면 기본 문구로 채움(무조건 4개 보장)
                while len(outs) < 4:
                    outs.append("정답에 도달하기 위한 중간 추론을 간략히 제시했습니다.")

                return outs[:4]

            except Exception as e:
                print(f"GPT API 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  → {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"  → 최대 재시도 횟수 초과, 기본값 4개 반환")
                    return ["정답에 도달하기 위한 중간 추론을 간략히 제시했습니다."] * 4

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

                # 변경: 4개 중간 추론 생성
                cot_results = self.generate_thinking_process_for_utterance(
                    utterances, j, label_info["label_kr"]
                )
                # cot_results는 길이 4의 리스트(문자열)임

                # 출력 형식에 맞게 구성
                output_item = {
                    "id": label_info['id'],
                    "label": label_info['label'],
                    "label_kr": label_info['label_kr'],
                    "cot": cot_results  # 4개 리스트로 저장
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
        print("CoT 데이터 생성 시작... (관점 없이 중간 추론 4개/발화)")

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
