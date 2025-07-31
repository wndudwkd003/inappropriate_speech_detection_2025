import json
import torch
from torch.utils.data import Dataset


# 클래스 밖으로 빼낸 상수와 함수
PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. \
    당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
    단, 동일한 문장을 절대 반복하지 마시오.'''

# question type별 instruction 정의
TYPE_INSTRUCTIONS = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        ),
        "교정형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        ),
        "선택형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        )
    }


def make_chat(inp):
    """입력 데이터를 채팅 형식으로 변환하는 함수"""

    # question type에 따른 instruction 선택
    instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

    # 기타 정보 생성 (question과 question_type 제외)
    other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

    # 기타 정보가 있는 경우에만 추가
    chat_parts = [instruction]
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))

    # 질문 추가
    chat_parts.append(f"[질문]\n{inp['question']}")

    # 최종 프롬프트 생성
    chat = "\n\n".join(chat_parts)

    return chat


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []

        with open(fname, "r") as f:
            data = json.load(f)

        for example in data:
            # 외부 함수 사용
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )

            # ① example["output"] 가 dict 면 내부 문자열을 꺼내고, 아니면 그대로
            raw_output = example.get("output", "")
            if isinstance(raw_output, dict):
                target_text = raw_output.get("answer", "")
            else:
                target_text = raw_output

            # ② answer 문자열이 있으면 EOS 토큰 추가
            if target_text:
                target_text = target_text + tokenizer.eos_token

            # ③ 토크나이즈
            target = tokenizer(
                target_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        # 각 샘플을 dict로 반환해야 Trainer가 올바르게 배치화(pad & stack)합니다.
        return {
            "input_ids": self.inp[idx],   # 1D 텐서([seq_len])
            "labels":    self.label[idx], # 1D 텐서([seq_len])
        }


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # 수정된 부분: torch.tensor() 제거
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# class DataCollatorForSupervisedDataset(object):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, instances):
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=-100
#         )

#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )
