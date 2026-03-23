from __future__ import annotations

import json
from pathlib import Path

from src.data.load import TrainExample


DEFAULT_SYSTEM_PROMPT = (
    "You are solving a reasoning task. "
    "Return the final answer in \\boxed{} format when possible. "
    "If the answer contains characters that make boxed output ambiguous, return the exact raw answer."
)


def can_safely_box(answer: str) -> bool:
    return not any(char in answer for char in "{}\\")


def build_assistant_target(answer: str) -> str:
    cleaned = answer.strip()
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        return cleaned
    if can_safely_box(cleaned):
        return f"\\boxed{{{cleaned}}}"
    return cleaned


def format_train_example(
    example: TrainExample,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, object]:
    assistant_text = build_assistant_target(example.answer)
    return {
        "id": example.example_id,
        "prompt": example.prompt,
        "answer": example.answer,
        "system": system_prompt,
        "user": example.prompt,
        "assistant": assistant_text,
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
