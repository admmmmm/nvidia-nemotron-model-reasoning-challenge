from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from src.data.synthetic_public_like import GENERATOR_BUILDERS
from src.data.synthetic_public_like.common import write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate public-style synthetic reasoning data.")
    parser.add_argument("--output-dir", required=True, help="Directory where JSONL files will be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument("--validation-size", type=float, default=0.05, help="Fraction of rows to place in val.jsonl.")
    parser.add_argument("--unit-conversion-count", type=int, default=0)
    parser.add_argument("--bit-manipulation-count", type=int, default=0)
    parser.add_argument("--cipher-count", type=int, default=0)
    parser.add_argument("--equation-numeric-count", type=int, default=0)
    parser.add_argument("--equation-symbolic-count", type=int, default=0)
    return parser.parse_args()


def _build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    count_by_category = {
        "unit_conversion": args.unit_conversion_count,
        "bit_manipulation": args.bit_manipulation_count,
        "cipher": args.cipher_count,
        "equation_numeric": args.equation_numeric_count,
        "equation_symbolic": args.equation_symbolic_count,
    }
    for category, count in count_by_category.items():
        builder = GENERATOR_BUILDERS[category]
        for index in range(count):
            rows.append(builder(rng, index))
    rng.shuffle(rows)
    return rows


def _split_rows(rows: list[dict[str, Any]], validation_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 1000)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * validation_size))
    val_count = min(max(val_count, 0), len(shuffled))
    return shuffled[val_count:], shuffled[:val_count]


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(args)
    train_rows, val_rows = _split_rows(rows, args.validation_size, args.seed)
    metadata_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "source": row["source"],
            "generator_family": row["generator_family"],
        }
        for row in rows
    ]

    write_jsonl(output_dir / "all.jsonl", rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "metadata.jsonl", metadata_rows)


if __name__ == "__main__":
    main()
