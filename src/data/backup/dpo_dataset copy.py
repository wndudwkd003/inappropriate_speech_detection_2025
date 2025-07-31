import json
from typing import Optional, Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class DPODataset(Dataset):
    """
    TRL DPOTrainer가 요구하는 컬럼(prompt, chosen, rejected)을 제공하는 Dataset.
    
    Args:
        fname (str): JSON(.json / .jsonl) 파일 경로
        tokenizer (PreTrainedTokenizerBase): 토크나이저
        max_length (int, optional): collator에서 사용할 max_length를 참고용으로 저장만. 실제 자르기는 collator가 합니다.
        add_system_prompt (bool): system prompt를 prompt 앞에 붙일지 여부
        use_chat_template (bool): HF chat template 사용 여부. False면 수동 문자열 결합.
    """

    PROMPT_SYSTEM = (
        "You are a helpful AI assistant. Please answer the user's questions kindly. "
        "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. "
        "단, 동일한 문장을 절대 반복하지 마시오."
    )

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
        ),
    }

    def __init__(
        self,
        fname: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        add_system_prompt: bool = True,
        use_chat_template: bool = True,
        eos_token: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_system_prompt = add_system_prompt
        self.use_chat_template = use_chat_template
        self.eos_token = eos_token #or tokenizer.eos_token

        # 파일 로딩
        if fname.endswith(".jsonl"):
            with open(fname, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f]
        else:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_data = data if isinstance(data, list) else data.get("data", [])

        # 포맷 검증
        required_keys = {"prompt", "chosen", "rejected"}
        for i, ex in enumerate(raw_data):
            if not required_keys.issubset(ex.keys()):
                raise ValueError(
                    f"Sample {i} is missing one of {required_keys}. Found keys: {list(ex.keys())}"
                )

        # 필요하다면 prompt 재가공 (예: question_type 있는 JSON → prompt로 합치기)
        # 여기서는 raw_data에 이미 prompt/chosen/rejected가 있다고 가정하지만,
        # 질문 타입별 instruction 로직을 쓰고 싶다면 아래처럼 다시 생성할 수 있음.
        self.samples: List[Dict[str, str]] = []
        for ex in raw_data:
            prompt_text = ex["prompt"]
            chosen = ex["chosen"]
            rejected = ex["rejected"]

            # eos 붙이기
            if self.eos_token:
                if not chosen.endswith(self.eos_token):
                    chosen += self.eos_token
                if not rejected.endswith(self.eos_token):
                    rejected += self.eos_token

            # system prompt & chat template 적용 (선택)
            if self.add_system_prompt:
                # chat template 사용: tokenize=False로 문자열만 받기
                if self.use_chat_template:
                    message = [
                        {"role": "system", "content": self.PROMPT_SYSTEM},
                        {"role": "user", "content": prompt_text},
                    ]
                    prompt_text = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    # 수동 결합
                    prompt_text = (
                        f"{self.PROMPT_SYSTEM}\n\n"
                        f"[질문]\n{prompt_text}\n\n"
                    )

            self.samples.append(
                {
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DataCollatorForDPODataset:
    """
    DPOTrainer에서 요구하는 데이터 콜레이터.
    """
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_multiple = pad_to_multiple_of

    def __call__(self, features):
        print("###############")
        print(features[:5])
        exit()
        prompts   = [f["prompt"]   for f in features]
        chosens   = [f["chosen"]   for f in features]
        rejecteds = [f["rejected"] for f in features]

        # 프롬프트만
        prompt_enc = self.tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )

        # 프롬프트 + chosen / rejected
        chosen_enc = self.tok(
            [p + c for p, c in zip(prompts, chosens)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )
        rejected_enc = self.tok(
            [p + r for p, r in zip(prompts, rejecteds)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )

        return {
            "prompt_input_ids":        prompt_enc["input_ids"],
            "prompt_attention_mask":   prompt_enc["attention_mask"],
            "chosen_input_ids":        chosen_enc["input_ids"],
            "chosen_attention_mask":   chosen_enc["attention_mask"],
            "rejected_input_ids":      rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }