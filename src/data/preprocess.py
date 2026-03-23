from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.load import TrainExample, load_train_examples, summarize_train_examples


def split_train_examples(
    examples: list[TrainExample],
    validation_size: float,
    seed: int,
) -> tuple[list[TrainExample], list[TrainExample]]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if len(examples) < 2:
        raise ValueError("Need at least two training examples to build a split.")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * validation_size))
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]
    if not train_examples:
        raise ValueError("Validation split is too large and leaves no training rows.")
    return train_examples, val_examples


def write_jsonl(path: str | Path, rows: list[dict[str, str]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local train/validation splits.")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--output-dir", default="data/splits/default")
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_train_examples(args.train_file)
    train_examples, val_examples = split_train_examples(
        examples=examples,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", [item.to_record() for item in train_examples])
    write_jsonl(output_dir / "val.jsonl", [item.to_record() for item in val_examples])

    metadata = {
        "train_file": str(args.train_file),
        "seed": args.seed,
        "validation_size": args.validation_size,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "summary": summarize_train_examples(examples),
    }
    write_jsonl(output_dir / "metadata.jsonl", [metadata])
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
