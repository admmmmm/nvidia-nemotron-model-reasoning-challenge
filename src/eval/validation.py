from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.eval.metric_local import load_predictions, score_predictions


def load_val_answers(path: str | Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = eval_json_line(line)
            answers[row["id"]] = row["answer"]
    return answers


def eval_json_line(line: str) -> dict[str, str]:
    import json

    row = json.loads(line)
    return {
        "id": str(row["id"]),
        "answer": str(row["answer"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate predictions against a held-out split.")
    parser.add_argument("--val-file", default="data/splits/default/val.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-column")
    parser.add_argument("--output-errors", default="outputs/logs/validation_errors.csv")
    return parser.parse_args()


def write_errors(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "gold_answer",
                "raw_prediction",
                "extracted_prediction",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows([row for row in rows if not row["is_correct"]])


def main() -> None:
    args = parse_args()
    gold_by_id = load_val_answers(args.val_file)
    predictions_by_id = load_predictions(args.predictions, args.prediction_column)
    report = score_predictions(gold_by_id, predictions_by_id)
    write_errors(args.output_errors, report["rows"])
    print(
        f"Validation accuracy: {report['accuracy']:.4f} "
        f"({report['correct']}/{report['total']})"
    )


if __name__ == "__main__":
    main()
