import json, torch, random, re
from typing import Any
from abc import ABC, abstractmethod
from src.configs.config_manager import ConfigManager

DEBUG = False  # 디버깅 모드 설정

OTHER_INFO_MAP = {
    "category": "카테고리",
    "topic_keyword": "키워드",
    "domain": "도메인",
    "question_type": "질문유형",
}

class BaseDataset(ABC):
    def __init__(
        self,
        fname: str,
        tokenizer,
        config_manager: ConfigManager,
        data_shuffle=False,
        task_type: str = "sft",  # ← task type 입력받기
        is_train: bool = True,
    ):
        self.fname = fname
        self.tokenizer = tokenizer
        self.config_manager = config_manager
        self.IGNORE_INDEX = -100
        self.data_shuffle = data_shuffle
        self.task_type = task_type
        self.max_seq_length = config_manager.model.max_seq_length
        self.is_train = is_train

        self.samples = []  # input_ids/labels 또는 prompt/chosen/rejected 통합 저장
        self.perspectives = [
            "대화에서 사용된 언어 표현 중 부적절한 단어나 표현(욕설, 비속어)이 발견되었는지?",
            "대화가 부적절한 발화의 경우 어떤 원인(차별, 혐오, 편향, 폄하 등)에서 문제가 되는지?",
            "대화의 흐름을 고려했을 때 각 발화자의 발언이 대화 당사자에게는 어떤 영향을주고, 제3자에게는 어떤 영향을 미치는지?",
            "대화의 전체 맥락과 발화 간의 상호작용을 고려하여 왜 각 발화가 적절 또는 부적절로 판단되었는지?",
        ]

        self._load_and_process_data()



    # def _load_and_process_data(self):
    #     with open(self.fname, "r") as f:
    #         data = json.load(f)

    #     if self.data_shuffle:
    #         print(f"Shuffling data... {len(data)} samples")
    #         random.shuffle(data)

    #     for samples in data:
    #         processed = self.process_sample(samples)
    #         if processed:
    #             self.samples.append(processed)

    def _load_and_process_data(self):
        with open(self.fname, "r") as f:
            data = json.load(f)

        if self.data_shuffle:
            print(f"Shuffling data... {len(data)} samples")
            random.shuffle(data)

        for samples in data:
            processed = self.process_sample(samples)
            if processed:
                # 리스트인 경우 extend 사용, 단일 항목인 경우 append 사용
                if isinstance(processed, list):
                    self.samples.extend(processed)
                else:
                    self.samples.append(processed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @abstractmethod
    def process_sample(
        self,
        sample: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        pass

def clean_retrieved_context(text):
    """
    검색된 컨텍스트에서 HTML 태그, 각주, 불필요한 문자들을 제거합니다.
    """
    if not text:
        return text

    # HTML 태그 제거 (단, <title>은 유지하되 태그만 제거)
    # <title>content</title> -> title: content 형태로 변환
    text = re.sub(r'<([^>]+?)>([^<]*?)</\1>', r'\1: \2', text)

    # 남은 HTML 태그들 제거
    text = re.sub(r'<[^>]*?>', '', text)

    # 각주 관련 패턴 제거
    text = re.sub(r'<templatestyles[^>]*?>', '', text)
    text = re.sub(r'<includeonly>[^<]*?</includeonly>', '', text)
    text = re.sub(r'각주\.\s*', '', text)

    # 특수 문자나 의미없는 패턴 제거
    text = re.sub(r'src\s*\"[^\"]*?\"', '', text)  # src="..." 제거
    text = re.sub(r'\/styles\.css', '', text)  # /styles.css 제거

    # 연속된 공백을 하나로 통일
    text = re.sub(r'\s+', ' ', text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text


def get_rag_context(sample: dict[str, Any], context_field: str = "retrieved_context", context_text="[관련 정보]") -> str:
    """RAG 사용 여부에 따라 context를 반환하는 함수. 예: [관련 정보] ~~~ """
    raw_context = sample.get(context_field, "")
    # RAG 컨텍스트 정리
    cleaned_context = clean_retrieved_context(raw_context)
    return f"{context_text} {cleaned_context}" if cleaned_context else raw_context



def check_limit_length(sample, limit_length: int) -> bool:
    # 질문 길이 제한 적용
    question_text = sample.get("input", {}).get("question", "")
    question_len = len(question_text.replace(" ", ""))  # 공백 제외

    if limit_length != -1 and question_len > limit_length:
        print(f"Skipping sample due to question length: {question_len} > {limit_length}")
        return True
    return False

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
