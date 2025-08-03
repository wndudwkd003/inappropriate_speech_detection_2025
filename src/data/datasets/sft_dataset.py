import torch
from src.data.prompt_manager import PromptManager
from src.data.datasets.base_dataset import BaseDataset, check_limit_length

DEBUG = False

class SFTDataset(BaseDataset):

    def make_full_chat_text(self, utterances):
        """전체 대화 텍스트 생성"""
        chat_text = []
        for idx, chat in enumerate(utterances):
            idx += 1
            speaker_seq = f"{chat['speaker_id']}_{idx}"
            chat_text.append(f"{speaker_seq}: {chat['form']}")
        return ", ".join(chat_text)

    def make_current_chat_text(self, utterance, utterance_idx):
        """현재 발화 텍스트 생성"""
        speaker_seq = f"{utterance['speaker_id']}_{utterance_idx + 1}"
        return f"{speaker_seq}: {utterance['form']}"

    def validate_and_convert_label(self, label):
        """label 검증 및 변환"""
        if label == "inappropriate":
            return "부적절" if self.config_manager.system.only_decode else 1
        elif label == "appropriate":
            return "적절" if self.config_manager.system.only_decode else 0
        else:
            print(f"Error: Invalid label '{label}'. Expected 'inappropriate' or 'appropriate'.")
            exit(1)

    def _build_user_content(self, full_chat_text: str, current_chat_text: str):
        perspective_text = ""
        if self.config_manager.system.perspective_count > 0:
            limited_perspectives = self.perspectives[:self.config_manager.system.perspective_count]
            if limited_perspectives:
                perspective_list = []
                for i, perspective in enumerate(limited_perspectives):
                    perspective_list.append(f"{i+1}. {perspective}")
                perspective_text = f"다음 관점들을 고려하세요: {' '.join(perspective_list)} "

        # 타입별 지시사항 가져오기
        type_instruction = PromptManager.get_type_instructions(self.config_manager.system.prompt_version)

        # 사용자 프롬프트 생성 (관점 + 타입 지시사항 + 전체 대화 + 현재 발화)
        user_content = f"{perspective_text}{type_instruction} 전체 대화: {full_chat_text} 현재 발화: {current_chat_text}"
        return user_content.strip()

    def _build_answer_text(self, converted_label, output):
        """
        only_decode=True일 때 학습용 정답 텍스트 생성.
        tokenizer가 없을 때는 eos 토큰을 붙이지 않음(순수 텍스트 반환).
        """
        is_cot = getattr(self.config_manager.system, "is_cot", False)
        pcnt = getattr(self.config_manager.system, "perspective_count", 0)

        if is_cot and pcnt > 0:
            cot_data = (output or {}).get("cot", [])
            limited_cot = cot_data[:pcnt]
            think_content = ""
            for cot_item in limited_cot:
                think_content += f"{cot_item}\n"
            core = f"<think>\n{think_content.strip()}</think>\n<answer>{converted_label}</answer>"
        else:
            core = str(converted_label)

        # tokenizer가 있을 때만 eos 사용
        if self.tokenizer is not None and getattr(self.tokenizer, "eos_token", None):
            return core + self.tokenizer.eos_token
        return core


    def process_sample(self, sample):
        # 질문 길이 제한 적용
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        utterances = sample["input"]["utterance"]
        outputs = sample.get("output", [])  # TEST 모드에서는 output이 없을 수 있음

        # TRAIN 모드에서는 발화 수와 출력 수가 일치해야 함
        if self.is_train and len(utterances) != len(outputs):
            print(f"Error: Mismatch between utterances ({len(utterances)}) and outputs ({len(outputs)})")
            return None

        samples = []
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)
        full_chat_text = self.make_full_chat_text(utterances)
        has_tokenizer = self.tokenizer is not None

        for idx, utterance in enumerate(utterances):
            current_chat_text = self.make_current_chat_text(utterance, idx)
            user_content = self._build_user_content(full_chat_text, current_chat_text)

            if self.config_manager.system.only_decode:
                # ------------------------
                # 생성형(SFT) 처리
                # ------------------------
                if self.is_train:
                    output = outputs[idx]
                    converted_label = self.validate_and_convert_label(output["label"])
                    answer_text = self._build_answer_text(converted_label, output)

                    if DEBUG:
                        print(f"Only decode mode: {self.config_manager.system.only_decode}")
                        print(f"Answer for utterance {idx + 1}: {answer_text}")

                    if has_tokenizer:
                        # chat template 우선
                        if getattr(self.tokenizer, "chat_template", None):
                            source = self.tokenizer.apply_chat_template(
                                [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": user_content}],
                                add_generation_prompt=True,
                                return_tensors="pt",
                                enable_thinking=False
                            )
                            source_ids = source["input_ids"][0] if isinstance(source, dict) else source[0]
                        else:
                            source = self.tokenizer(
                                f"{system_prompt} {user_content}",
                                return_tensors="pt",
                                add_special_tokens=True
                            )
                            source_ids = source["input_ids"][0]

                        target = self.tokenizer(
                            answer_text,
                            return_attention_mask=False,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )
                        target["input_ids"] = target["input_ids"].type(torch.int64)

                        input_ids = torch.concat((source_ids, target["input_ids"][0]))
                        labels = torch.concat((torch.LongTensor([self.IGNORE_INDEX] * source_ids.shape[0]),
                                               target["input_ids"][0]))

                        samples.append({
                            "input_ids": input_ids,
                            "labels": labels
                        })
                    else:
                        # 토크나이저 없음: 텍스트만 반환
                        samples.append({
                            "source_text": f"{system_prompt} {user_content}",
                            "target_text": answer_text
                        })
                else:
                    # TEST 모드: 입력만 준비
                    if has_tokenizer:
                        if getattr(self.tokenizer, "chat_template", None):
                            source = self.tokenizer.apply_chat_template(
                                [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": user_content}],
                                add_generation_prompt=True,
                                return_tensors="pt",
                                enable_thinking=False
                            )
                            source_ids = source["input_ids"][0] if isinstance(source, dict) else source[0]
                        else:
                            source = self.tokenizer(
                                f"{system_prompt} {user_content}",
                                return_tensors="pt",
                                add_special_tokens=True
                            )
                            source_ids = source["input_ids"][0]

                        samples.append({
                            "input_ids": source_ids,
                            "attention_mask": torch.ones_like(source_ids),
                            "document_id": sample["id"],
                            "utterance_id": utterance["id"],
                            "utterance_idx": idx
                        })
                    else:
                        samples.append({
                            "system_prompt": system_prompt,
                            "user_content": user_content,
                            "document_id": sample["id"],
                            "utterance_id": utterance["id"],
                            "utterance_idx": idx
                        })

            else:
                # ------------------------
                # 분류 모델 처리(only_decode=False)
                # ------------------------
                input_text = f"{system_prompt} {user_content}"

                if self.is_train:
                    output = outputs[idx]
                    converted_label = self.validate_and_convert_label(output["label"])

                if DEBUG:
                    print(f"Input text for utterance {idx + 1}: {input_text}")
                    if self.is_train:
                        print(f"Label for utterance {idx + 1}: {converted_label}")

                if has_tokenizer:
                    tokenized = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=self.max_seq_length,
                        padding=False,
                        return_tensors="pt"
                    )
                    sample_data = {
                        "input_ids": tokenized["input_ids"][0],
                        "attention_mask": tokenized["attention_mask"][0],
                        "document_id": sample["id"],
                        "utterance_id": utterance["id"],
                        "utterance_idx": idx
                    }
                    if self.is_train:
                        sample_data["labels"] = torch.tensor(converted_label, dtype=torch.long)
                    samples.append(sample_data)
                else:
                    # 토크나이저 없음: 텍스트만 반환
                    sample_data = {
                        "system_prompt": system_prompt,
                        "user_content": user_content,
                        "document_id": sample["id"],
                        "utterance_id": utterance["id"],
                        "utterance_idx": idx
                    }
                    if self.is_train:
                        sample_data["label"] = converted_label  # int 라벨
                    samples.append(sample_data)

        return samples
