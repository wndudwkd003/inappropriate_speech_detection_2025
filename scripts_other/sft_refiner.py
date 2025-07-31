import json
import os

class ThinkTagAdder:
    def __init__(self):
        self.data_dir = "data/other/kowiki_2025_results_for_cot"

    def get_think_content(self, question_type):
        """질문 유형에 따른 think 내용 반환"""
        if question_type == "서술형":
            return "이 문제는 서술형이다. 중요한 정보를 포함하여 완성된 서술형으로 작성해야 한다."
        elif question_type == "단답형":
            return "이 문제는 단답형 문제이다. 정확한 단어로 대답해야 한다."
        else:
            # 선다형이나 기타의 경우 기존 상태 유지
            return None

    def update_answer_with_think(self, answer, question_type):
        """기존 answer에 think 태그 추가"""
        think_content = self.get_think_content(question_type)

        if think_content is None:
            # 선다형이나 기타의 경우 원본 유지
            return answer

        # 기존에 think 태그가 있는지 확인
        if '<think>' in answer and '</think>' in answer:
            # 이미 think 태그가 있으면 내용만 업데이트
            import re
            pattern = r'<think>(.*?)</think>'
            new_think = f"<think>\n{think_content}\n</think>"
            updated_answer = re.sub(pattern, new_think, answer, flags=re.DOTALL)
            return updated_answer
        else:
            # think 태그가 없으면 추가
            if '<answer>' in answer:
                # answer 태그가 있는 경우
                answer_start = answer.find('<answer>')
                new_answer = f"<think>\n{think_content}\n</think>{answer[answer_start:]}"
                return new_answer
            else:
                # answer 태그가 없는 경우 전체를 answer로 감싸고 think 추가
                return f"<think>\n{think_content}\n</think><answer>\n{answer}\n</answer>"

    def process_file(self, filename):
        """개별 파일 처리"""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"파일이 존재하지 않습니다: {filepath}")
            return

        print(f"{filename} 처리 시작...")

        # 파일 로드
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated_count = 0
        skipped_count = 0

        for i, item in enumerate(data):
            question_type = item["input"]["question_type"]
            current_answer = item["output"]["answer"]

            # 서술형이나 단답형인 경우에만 처리
            if question_type in ["서술형", "단답형"]:
                updated_answer = self.update_answer_with_think(current_answer, question_type)

                if updated_answer != current_answer:
                    item["output"]["answer"] = updated_answer
                    updated_count += 1
                    print(f"  → {i+1}: {question_type} 업데이트 완료")
                else:
                    print(f"  → {i+1}: {question_type} 이미 적절한 think 태그 존재")
            else:
                skipped_count += 1
                print(f"  → {i+1}: {question_type} 스킵")

        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{filename} 처리 완료: {updated_count}개 업데이트, {skipped_count}개 스킵")

    def run(self):
        """전체 실행"""
        print("Think 태그 추가 작업 시작...")

        # dev.json과 train.json 처리
        for filename in ["dev.json", "train.json"]:
            if os.path.exists(os.path.join(self.data_dir, filename)):
                print(f"\n{filename} 처리 시작...")
                self.process_file(filename)
            else:
                print(f"\n{filename} 파일이 존재하지 않습니다.")

        print("\n모든 작업 완료!")

# 사용 예시
if __name__ == "__main__":
    adder = ThinkTagAdder()
    adder.run()
