from enum import Enum

class PromptVersion(Enum):
    V0 = "nothing"
    V1 = "original"
    V2 = "soft_cot"

class PromptManager:
    # 시스템 프롬프트
    SYSTEM_PROMPTS = {
        PromptVersion.V0: "",
        PromptVersion.V1: (
            "당신은 대화에서 부적절한 발언을 탐지할 수 있는 어시스턴트 AI입니다. "
            "전체 대화 맥락을 바탕으로 현재 발화가 '적절'인지 또는 '부적절'인지 판단할 수 있습니다. "
            "사용자의 지시사항을 충실히 이행하고 정확하게 답변하세요. 또한, 질문과 동일한 문장을 반복하지 마세요."
        ),
        PromptVersion.V2: (
            "당신은 대화에서 부적절한 발언을 탐지할 수 있는 어시스턴트 AI입니다. "
            "전체 대화 맥락을 바탕으로 현재 발화가 '적절'인지 또는 '부적절'인지 판단할 수 있습니다. "
            "사용자의 지시사항을 충실히 이행하고 정확하게 답변하세요. 또한, 질문과 동일한 문장을 반복하지 마세요."
        ),

    }

    # 질문 타입별 지시 정의
    TYPE_INSTRUCTIONS = {
        PromptVersion.V0: "",
        PromptVersion.V1: "지시사항: 현재 발화가 '적절'인지 또는 '부적절'인지 판단하여 답변하세요.",
        PromptVersion.V2: (
            "지시사항: 주어진 관점에 따라 논리적인 연결고리를 통해 정답으로 귀결되는 추론을 할 수 있습니다. "
            "각 관점에 따른 단계적 생각의 연결 고리는 <think> </think> 태그로 감싸세요. "
            "최종 답변은 현재 발화가 '적절'인지 또는 '부적절'인지 <answer> </answer> 태그로 감싸서 답변하세요."
        )
    }

    @classmethod
    def get_system_prompt(cls, version: PromptVersion) -> str:
        """지정된 버전의 시스템 프롬프트 반환"""
        return cls.SYSTEM_PROMPTS.get(version, cls.SYSTEM_PROMPTS[PromptVersion.V1])

    @classmethod
    def get_type_instructions(cls, version: PromptVersion) -> str:
        """지정된 버전의 타입별 instruction 반환"""
        return cls.TYPE_INSTRUCTIONS.get(version, cls.TYPE_INSTRUCTIONS[PromptVersion.V1])

    @classmethod
    def get_instruction_for_type(cls, version: PromptVersion, question_type: str) -> str:
        """특정 버전과 질문 타입에 대한 instruction 반환"""
        type_instructions = cls.get_type_instructions(version)
        return type_instructions.get(question_type, "")
