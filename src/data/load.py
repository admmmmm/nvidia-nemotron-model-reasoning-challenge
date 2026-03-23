from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TRAIN_COLUMNS = ("id", "prompt", "answer")
TEST_COLUMNS = ("id", "prompt")


class SchemaError(ValueError):
    """Raised when a CSV file does not match the expected schema."""


@dataclass(frozen=True)
class TrainExample:
    example_id: str
    prompt: str
    answer: str

    def to_record(self) -> dict[str, str]:
        return {
            "id": self.example_id,
            "prompt": self.prompt,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class TestExample:
    example_id: str
    prompt: str

    def to_record(self) -> dict[str, str]:
        return {
            "id": self.example_id,
            "prompt": self.prompt,
        }


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f"{path} is empty or missing a header row.")
        rows = list(reader)
    return list(reader.fieldnames), rows


def validate_columns(path: Path, fieldnames: Iterable[str], expected: Iterable[str]) -> None:
    actual_tuple = tuple(fieldnames)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        raise SchemaError(
            f"{path} has columns {actual_tuple}, expected {expected_tuple}."
        )


def load_train_examples(path: str | Path) -> list[TrainExample]:
    csv_path = Path(path)
    fieldnames, rows = _read_csv_rows(csv_path)
    validate_columns(csv_path, fieldnames, TRAIN_COLUMNS)
    return [
        TrainExample(
            example_id=row["id"].strip(),
            prompt=row["prompt"].strip(),
            answer=row["answer"].strip(),
        )
        for row in rows
    ]


def load_test_examples(path: str | Path) -> list[TestExample]:
    csv_path = Path(path)
    fieldnames, rows = _read_csv_rows(csv_path)
    validate_columns(csv_path, fieldnames, TEST_COLUMNS)
    return [
        TestExample(
            example_id=row["id"].strip(),
            prompt=row["prompt"].strip(),
        )
        for row in rows
    ]


def summarize_train_examples(examples: Iterable[TrainExample]) -> dict[str, float]:
    rows = list(examples)
    prompt_lengths = [len(item.prompt) for item in rows]
    answer_lengths = [len(item.answer) for item in rows]
    count = len(rows)

    def _avg(values: list[int]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    return {
        "count": float(count),
        "avg_prompt_chars": _avg(prompt_lengths),
        "avg_answer_chars": _avg(answer_lengths),
        "max_prompt_chars": float(max(prompt_lengths) if prompt_lengths else 0),
        "max_answer_chars": float(max(answer_lengths) if answer_lengths else 0),
    }


def examples_to_records(examples: Iterable[TrainExample | TestExample]) -> list[dict[str, str]]:
    return [item.to_record() for item in examples]
