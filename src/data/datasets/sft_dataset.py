import torch
from src.data.prompt_manager import PromptManager
from src.data.base_dataset import BaseDataset, make_chat, check_limit_length

DEBUG = False



class SFTDataset(BaseDataset):

    def make_chat_for_app_inapp(self, sample):
        chat_text = []
        speaker_sequence = []
        for idx, chat in enumerate(sample):
            idx += 1
            speaker_seq = f"{chat['speaker_id']}_{idx}"
            chat_text.append(f"{speaker_seq}: {chat['form']}")
        input_text = ", ".join(chat_text)
        if DEBUG: print(f"Input Text: {input_text}")  # 디버깅용 출력
        return input_text, speaker_sequence

    def make_chat_answer_for_app_inapp(self, sample, speaker_sequence):
        answer_text = []
        for idx, answer in enumerate(sample):
            answer_text.append(f"{speaker_sequence[idx]}: {answer['label']}")
        output_text = ", ".join(answer_text)
        if DEBUG: print(f"Output Text: {output_text}")  # 디버깅용 출력
        return output_text

    def process_sample(self, sample):
        # 질문 길이 제한 적용
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        # 시스템 프롬프트 가져오기
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)

        # 사용자 프롬프트 생성
        input_text, speaker_sequence = self.make_chat_for_app_inapp(
            sample["utterance"],
        )

        # 메시지 구성
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ]

        if DEBUG: print(message)

        # 소스 토크나이즈
        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )

        answer = self.make_chat_answer_for_app_inapp(
            sample["output"],
            speaker_sequence
        )

        if not answer:
            return None

        # EOS 토큰 추가
        answer = answer + self.tokenizer.eos_token

        if DEBUG: print(answer)  # 디버깅용 출력

        # 타겟 토크나이즈
        target = self.tokenizer(
            answer,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        target["input_ids"] = target["input_ids"].type(torch.int64)

        # input_ids = torch.concat((source[0], target["input_ids"][0]))
        if isinstance(source, dict):
            source_ids = source["input_ids"][0]
        else:
            source_ids = source[0]

        input_ids = torch.concat((source_ids, target["input_ids"][0]))

        labels = torch.concat((torch.LongTensor([self.IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))

        return {
            "input_ids": input_ids,
            "labels": labels
        }


