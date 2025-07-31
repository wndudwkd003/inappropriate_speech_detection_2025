import json
import os
import shutil
import openai
import time
import yaml

class CoTDataGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.input_dir = "data/other/kowiki_2025_results"
        self.output_dir = "data/other/kowiki_2025_results_for_cot"

    def setup_output_directory(self):
        """출력 디렉토리 설정 및 test.json 복사"""
        os.makedirs(self.output_dir, exist_ok=True)

        # test.json 그대로 복사
        test_src = os.path.join(self.input_dir, "test.json")
        test_dst = os.path.join(self.output_dir, "test.json")
        if os.path.exists(test_src):
            shutil.copy2(test_src, test_dst)
            print(f"test.json 복사 완료: {test_dst}")

    def generate_thinking_process(self, category, domain, question_type, topic_keyword, question, retrieved_context, correct_answer, max_retries=3):
        """GPT API를 사용해 선다형 질문의 간단한 추론 과정 생성"""
        prompt = f"""다음 질문에 대해 간단하고 논리적인 추론 과정을 2-3문장으로 작성해주세요.

카테고리: {category}
도메인: {domain}
질문 유형: {question_type}
주제 키워드: {topic_keyword}
질문: {question}
검색된 내용: {retrieved_context}
정답: {correct_answer}

요구사항: 2~3문장 이내로 간결하지만 정답에 도달하기 위한 핵심 논리적 연결 고리를 출력해야합니다. 검색된 내용에는 질문 및 답변과 연관된 내용이 있을 수 있습니다. 검색된 내용과 연관지어서 설명하세요. 만약, 관련 정보가 없으면 '검색 내용에 직접적인 정보가 없다'라는 내용을 추가하고 스스로 생각하는 과정을 적으세요.

질문에 대한 답변을 하는것이 아니라 추론 과정만 작성하세요."""

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=[
                        {"role": "system", "content": "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 그리고 논리적인 추론 과정을 출력하는 역할을 수행합니다. 핵심만 간결하게 설명하세요. 한국어 문법에 자연스럽게 답하시고 생각하는 과정은 존댓말을 사용하지 마세요."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=512
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"GPT API 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2초, 4초, 6초 대기
                    print(f"  → {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"  → 최대 재시도 횟수 초과, 기본값 반환")
                    return f"검색 내용을 분석하여 {correct_answer}이 정답임을 확인했다."

    def format_cot_answer(self, thinking_process, correct_answer):
        """추론 과정과 정답을 CoT 형태로 포맷팅"""
        return f"<think>\n{thinking_process}\n</think><answer>\n{correct_answer}\n</answer>"

    def format_simple_answer(self, correct_answer):
        """선다형이 아닌 경우 빈 think 태그와 answer 태그로 포맷팅"""
        return f"<think>\n\n</think><answer>\n{correct_answer}\n</answer>"

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
            # 기존 파일이 있으면 로드
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            # 파일이 없으면 빈 리스트로 시작
            existing_data = []

        # 새 항목들 추가
        existing_data.extend(new_items)

        # 파일에 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def process_file(self, filename):
        """개별 파일 처리"""
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)

        if not os.path.exists(input_path):
            print(f"파일이 존재하지 않습니다: {input_path}")
            return

        # 기존에 저장된 ID들 로드
        existing_ids = self.load_existing_ids(output_path)
        print(f"기존 저장된 데이터: {len(existing_ids)}개")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_processed = 0
        skipped_count = 0

        for i, item in enumerate(data):
            print(f"{filename} 처리 중: {i+1}/{len(data)}")

            # 이미 처리된 데이터면 건너뛰기
            if item['id'] in existing_ids:
                skipped_count += 1
                print(f"  → 이미 처리됨: {item['id']} (스킵)")
                continue

            # 원본 데이터에서 필요한 정보 추출
            category = item["input"]["category"]
            domain = item["input"]["domain"]
            question_type = item["input"]["question_type"]
            topic_keyword = item["input"]["topic_keyword"]
            question = item["input"]["question"]
            retrieved_context = item["input"]["retrieved_context"]
            correct_answer = item["output"]["answer"]

            # 선다형인 경우 CoT 형태로, 아닌 경우 빈 think 태그와 answer 태그로 처리
            if question_type == "선다형":
                # GPT로 간단한 추론 과정 생성
                thinking_process = self.generate_thinking_process(
                    category, domain, question_type, topic_keyword,
                    question, retrieved_context, correct_answer
                )
                # CoT 형태로 포맷팅
                formatted_answer = self.format_cot_answer(thinking_process, correct_answer)
                print(f"  → 선다형: CoT 형태로 처리")
            else:
                # 빈 think 태그와 answer 태그로 포맷팅
                formatted_answer = self.format_simple_answer(correct_answer)
                print(f"  → {question_type}: 빈 think 태그로 처리")

            # 데이터 생성
            cot_item = {
                "id": item['id'],
                "input": item["input"].copy(),
                "output": {
                    "answer": formatted_answer
                }
            }

            # 즉시 파일에 저장
            self.append_to_json_file(output_path, [cot_item])
            total_processed += 1

            print(f"  → 저장 완료: {total_processed}개 데이터")

            # 선다형인 경우에만 API 호출 제한을 위한 딜레이
            if question_type == "선다형":
                time.sleep(0.1)

        print(f"{filename} 처리 완료: 전체 {len(data)}개 중 {total_processed}개 새로 처리, {skipped_count}개 스킵")

    def run(self):
        """전체 실행"""
        print("CoT 데이터 생성 시작...")

        # 출력 디렉토리 설정
        self.setup_output_directory()

        # dev.json과 train.json 처리
        for filename in ["dev.json", "train.json"]:
            print(f"\n{filename} 처리 시작...")
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
