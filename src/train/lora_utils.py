from __future__ import annotations

from peft import LoraConfig


NEMOTRON_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def choose_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "nemotron" in lowered:
        return NEMOTRON_TARGET_MODULES
    if "qwen" in lowered:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def build_lora_config(
    model_name: str,
    rank: int,
    alpha: int,
    dropout: float,
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=choose_target_modules(model_name),
    )
