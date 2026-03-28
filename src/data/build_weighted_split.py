from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.preprocess import write_jsonl


DEFAULT_STATUS_WEIGHTS = {
    "solved": 1,
    "partial": 2,
    "unsolved": 4,
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_generation_status(row: dict[str, Any]) -> str:
    if bool(row.get("latest_correct")):
        return "solved"
    if bool(row.get("any_correct")):
        return "partial"
    return "unsolved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a weighted training split using base-model outcome metadata on the official public train set."
    )
    parser.add_argument("--train-split", default="data/splits/default/train.jsonl")
    parser.add_argument("--val-split", default="data/splits/default/val.jsonl")
    parser.add_argument("--generation-file", default="data/external/nemotron_huikang_dev/generation.jsonl")
    parser.add_argument("--problems-file", default="data/external/nemotron_huikang_dev/problems.jsonl")
    parser.add_argument("--output-dir", default="data/splits/boxed_weighted_v1")
    parser.add_argument("--solved-weight", type=int, default=DEFAULT_STATUS_WEIGHTS["solved"])
    parser.add_argument("--partial-weight", type=int, default=DEFAULT_STATUS_WEIGHTS["partial"])
    parser.add_argument("--unsolved-weight", type=int, default=DEFAULT_STATUS_WEIGHTS["unsolved"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def attach_status(
    rows: list[dict[str, Any]],
    generation_by_id: dict[str, dict[str, Any]],
    problem_by_id: dict[str, dict[str, Any]],
    weights: dict[str, int],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        generation_row = generation_by_id.get(str(row["id"]), {})
        problem_row = problem_by_id.get(str(row["id"]), {})
        status = derive_generation_status(generation_row) if generation_row else "unknown"
        category = problem_row.get("category")
        item["base_model_status"] = status
        item["base_model_any_correct"] = bool(generation_row.get("any_correct", False))
        item["base_model_latest_correct"] = bool(generation_row.get("latest_correct", False))
        item["base_model_latest_extracted"] = generation_row.get("latest_extracted")
        if category is not None:
            item["category"] = category
        item["resample_weight"] = int(weights.get(status, 1))
        enriched.append(item)
    return enriched


def expand_weighted_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        weight = max(1, int(row.get("resample_weight", 1)))
        for repeat_index in range(weight):
            item = dict(row)
            item["repeat_index"] = repeat_index
            expanded.append(item)
    random.Random(seed).shuffle(expanded)
    return expanded


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train_split)
    val_rows = load_jsonl(args.val_split)
    generation_rows = load_jsonl(args.generation_file)
    problem_rows = load_jsonl(args.problems_file)

    generation_by_id = {str(row["id"]): row for row in generation_rows}
    problem_by_id = {str(row["id"]): row for row in problem_rows}
    weights = {
        "solved": args.solved_weight,
        "partial": args.partial_weight,
        "unsolved": args.unsolved_weight,
    }

    enriched_train = attach_status(
        train_rows,
        generation_by_id=generation_by_id,
        problem_by_id=problem_by_id,
        weights=weights,
    )
    enriched_val = attach_status(
        val_rows,
        generation_by_id=generation_by_id,
        problem_by_id=problem_by_id,
        weights=weights,
    )
    weighted_train = expand_weighted_rows(enriched_train, seed=args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", weighted_train)
    write_jsonl(output_dir / "val.jsonl", enriched_val)

    train_status_counts = Counter(row.get("base_model_status", "unknown") for row in enriched_train)
    val_status_counts = Counter(row.get("base_model_status", "unknown") for row in enriched_val)
    weighted_status_counts = Counter(row.get("base_model_status", "unknown") for row in weighted_train)

    metadata = {
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "generation_file": str(args.generation_file),
        "problems_file": str(args.problems_file),
        "weights": weights,
        "seed": args.seed,
        "train_count_original": len(enriched_train),
        "train_count_weighted": len(weighted_train),
        "val_count": len(enriched_val),
        "train_status_counts": dict(train_status_counts),
        "train_status_counts_weighted": dict(weighted_status_counts),
        "val_status_counts": dict(val_status_counts),
    }
    write_jsonl(output_dir / "metadata.jsonl", [metadata])
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
