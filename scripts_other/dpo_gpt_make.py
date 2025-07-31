import json
import os
import shutil
import openai
import time

class DPODataGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.input_dir = "data/other/kowiki_2025_results"
        self.output_dir = "data/other/kowiki_2025_results_for_dpo"

    def setup_output_directory(self):
        """출력 디렉토리 설정 및 test.json 복사"""
        os.makedirs(self.output_dir, exist_ok=True)

        # test.json 그대로 복사
        test_src = os.path.join(self.input_dir, "test.json")
        test_dst = os.path.join(self.output_dir, "test.json")
        if os.path.exists(test_src):
            shutil.copy2(test_src, test_dst)
            print(f"test.json 복사 완료: {test_dst}")

    def generate_single_wrong_answer(self, category, domain, question_type, topic_keyword, question, correct_answer):
        """GPT API를 사용해 틀린 답변 1개 생성"""
        prompt = f"""DPO 학습을 위한 거절 답변을 만들려고 합니다. 다음 질문에 대해 그럴듯하지만 틀린 답변 1개를 생성해주세요.

카테고리: {category}
도메인: {domain}
질문 유형: {question_type}
주제 키워드: {topic_keyword}
질문: {question}
정답: {correct_answer}

요구사항:
1. 정답과는 다르지만 그럴듯한 답변
2. 해당 도메인/카테고리와 관련된 용어 사용
3. 간단명료하게 (정답과 비슷한 길이)
4. 답변만 제공 (추가 설명 없이)"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {"role": "system", "content": "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 주어진 질문에 대해 그럴듯하지만 틀린 답변을 생성해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"GPT API 호출 오류: {e}")
            # 기본 틀린 답변 반환
            return f"오답_{topic_keyword}"

    def generate_wrong_answers(self, category, domain, question_type, topic_keyword, question, correct_answer):
        """GPT API를 두 번 호출해서 틀린 답변 2개 생성"""
        # 첫 번째 틀린 답변 생성
        rejected_1 = self.generate_single_wrong_answer(
            category, domain, question_type, topic_keyword, question, correct_answer
        )

        # API 호출 간격
        time.sleep(0.1)

        # 두 번째 틀린 답변 생성
        rejected_2 = self.generate_single_wrong_answer(
            category, domain, question_type, topic_keyword, question, correct_answer
        )

        return rejected_1, rejected_2

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

        # 출력 파일이 이미 있으면 삭제 (새로 시작)
        if os.path.exists(output_path):
            os.remove(output_path)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_processed = 0

        for i, item in enumerate(data):
            print(f"{filename} 처리 중: {i+1}/{len(data)}")

            # 원본 데이터에서 필요한 정보 추출
            category = item["input"]["category"]
            domain = item["input"]["domain"]
            question_type = item["input"]["question_type"]
            topic_keyword = item["input"]["topic_keyword"]
            question = item["input"]["question"]
            correct_answer = item["output"]["answer"]

            # GPT로 틀린 답변 2개 생성 (두 번 호출)
            rejected_1, rejected_2 = self.generate_wrong_answers(
                category, domain, question_type, topic_keyword, question, correct_answer
            )

            # 첫 번째 버전 데이터 생성
            item_v1 = {
                "id": f"{item['id']}_v1",
                "input": item["input"].copy(),
                "output": {
                    "answer": correct_answer,
                    "rejected": rejected_1
                }
            }

            # 두 번째 버전 데이터 생성
            item_v2 = {
                "id": f"{item['id']}_v2",
                "input": item["input"].copy(),
                "output": {
                    "answer": correct_answer,
                    "rejected": rejected_2
                }
            }

            # 즉시 파일에 저장
            self.append_to_json_file(output_path, [item_v1, item_v2])
            total_processed += 2

            print(f"  → 저장 완료: {total_processed}개 데이터")

            # API 호출 제한을 위한 딜레이
            time.sleep(0.1)

        print(f"{filename} 처리 완료: {len(data)} → {total_processed}개 데이터")

    def run(self):
        """전체 실행"""
        print("DPO 데이터 생성 시작...")

        # 출력 디렉토리 설정
        self.setup_output_directory()

        # dev.json과 train.json 처리
        for filename in ["dev.json", "train.json"]:
            print(f"\n{filename} 처리 시작...")
            self.process_file(filename)

        print("\n모든 작업 완료!")

# 사용 예시
if __name__ == "__main__":
    import yaml
    # OpenAI API 키 설정 (실제 사용시 환경변수나 별도 파일에서 로드)
    with open("src/configs/tokens/token.yaml", 'r', encoding='utf-8') as f:
        tokens = yaml.safe_load(f)
    API_KEY = tokens["open_ai_token"]

    generator = DPODataGenerator(API_KEY)
    generator.run()
