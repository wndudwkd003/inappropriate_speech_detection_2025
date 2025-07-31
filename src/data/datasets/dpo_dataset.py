from src.data.base_dataset import BaseDataset, make_chat, check_limit_length
from src.data.prompt_manager import PromptManager

DEBUG = False

class DPODataset(BaseDataset):
    """
    TRL DPOTrainer가 요구하는 컬럼(prompt, chosen, rejected)을 제공하는 Dataset.

    Args:
        다시 작성 예정
    """
    def process_sample(self, sample):
        # 질문 길이 제한 적용
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)
        user_prompt = make_chat(sample["input"], config_manager=self.config_manager)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
        )

        chosen_text = sample["output"].get("answer", "").strip()
        rejected_text = sample["output"].get("rejected", "").strip()

        if not chosen_text or not rejected_text:
            return None

        # eos 추가 (선택)
        eos = self.tokenizer.eos_token or ""
        chosen_text += eos
        rejected_text += eos

        result = {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

        return result
