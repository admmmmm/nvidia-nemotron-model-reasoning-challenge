from __future__ import annotations

import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

from src.eval.answer_extract import extract_answer


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def parse_numeric(text: str) -> float | None:
    cleaned = normalize_text(text).rstrip(".")
    if not cleaned:
        return None
    try:
        if "/" in cleaned and cleaned.count("/") == 1:
            return float(Fraction(cleaned))
        return float(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(gold: str, predicted: str, tolerance: float = 1e-2) -> bool:
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(predicted)

    if gold_norm == pred_norm:
        return True

    gold_value = parse_numeric(gold_norm)
    pred_value = parse_numeric(pred_norm)
    if gold_value is None or pred_value is None:
        return False

    if math.isclose(gold_value, pred_value, rel_tol=tolerance, abs_tol=tolerance):
        return True
    return False


def load_ground_truth(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["id"].strip()] = row["answer"].strip()
    return mapping


def load_predictions(path: str | Path, prediction_column: str | None) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Predictions file is missing a header row.")

        candidate_columns = [prediction_column] if prediction_column else []
        candidate_columns.extend(["prediction", "prediction_text", "response", "output"])
        resolved_column = next(
            (column for column in candidate_columns if column and column in reader.fieldnames),
            None,
        )
        if resolved_column is None:
            raise ValueError(
                "Could not find a prediction column. "
                f"Available columns: {reader.fieldnames}"
            )

        predictions: dict[str, str] = {}
        for row in reader:
            predictions[row["id"].strip()] = row[resolved_column]
    return predictions


def score_predictions(
    gold_by_id: dict[str, str],
    predictions_by_id: dict[str, str],
) -> dict[str, object]:
    scored_rows: list[dict[str, object]] = []
    correct = 0

    for example_id, gold_answer in gold_by_id.items():
        raw_prediction = predictions_by_id.get(example_id, "")
        extracted_prediction = extract_answer(raw_prediction)
        is_correct = answers_match(gold_answer, extracted_prediction)
        correct += int(is_correct)
        scored_rows.append(
            {
                "id": example_id,
                "gold_answer": gold_answer,
                "raw_prediction": raw_prediction,
                "extracted_prediction": extracted_prediction,
                "is_correct": is_correct,
            }
        )

    total = len(gold_by_id)
    accuracy = (correct / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "rows": scored_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local predictions against train.csv.")
    parser.add_argument("--ground-truth", default="train.csv")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-column")
    parser.add_argument("--output-errors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_by_id = load_ground_truth(args.ground_truth)
    predictions_by_id = load_predictions(args.predictions, args.prediction_column)
    report = score_predictions(gold_by_id, predictions_by_id)

    if args.output_errors:
        output_path = Path(args.output_errors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrong_rows = [row for row in report["rows"] if not row["is_correct"]]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
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
            writer.writerows(wrong_rows)

    print(
        json.dumps(
            {
                "accuracy": report["accuracy"],
                "correct": report["correct"],
                "total": report["total"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
