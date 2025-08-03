from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: int|float = 0
    target_modules: str|list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bias: str = "none"
    init_lora_weights: str|bool = True
    use_dora: bool = False
    use_rslora: bool = False

