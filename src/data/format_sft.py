from __future__ import annotations

import json
from pathlib import Path

from src.data.load import TrainExample


DEFAULT_SYSTEM_PROMPT = (
    "You are solving a reasoning task. "
    "The final line must contain the final answer. "
    "Return the final answer in \\boxed{} format when possible. "
    "Do not write anything after the final boxed answer. "
    "If the answer contains characters that make boxed output ambiguous, return the exact raw answer on the final line."
)

DEFAULT_ASSISTANT_TARGET_STYLE = "boxed_final_line"


def can_safely_box(answer: str) -> bool:
    return not any(char in answer for char in "{}\\")


def _build_boxed_answer(answer: str) -> str:
    cleaned = answer.strip()
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        return cleaned
    if can_safely_box(cleaned):
        return f"\\boxed{{{cleaned}}}"
    return cleaned


def build_assistant_target(answer: str, style: str = DEFAULT_ASSISTANT_TARGET_STYLE) -> str:
    cleaned = answer.strip()
    boxed = _build_boxed_answer(cleaned)
    normalized_style = style.strip().lower()

    if normalized_style == "raw":
        return cleaned
    if normalized_style == "boxed_only":
        return boxed
    if normalized_style == "boxed_final_line":
        if boxed == cleaned:
            return cleaned
        return f"Final answer:\n{boxed}"
    raise ValueError(
        f"Unsupported assistant target style: {style!r}. "
        "Expected one of: raw, boxed_only, boxed_final_line."
    )


def format_train_example(
    example: TrainExample,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    assistant_target_style: str = DEFAULT_ASSISTANT_TARGET_STYLE,
) -> dict[str, object]:
    assistant_text = build_assistant_target(example.answer, style=assistant_target_style)
    return {
        "id": example.example_id,
        "prompt": example.prompt,
        "answer": example.answer,
        "system": system_prompt,
        "user": example.prompt,
        "assistant": assistant_text,
        "assistant_target_style": assistant_target_style,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def load_jsonl_records(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
