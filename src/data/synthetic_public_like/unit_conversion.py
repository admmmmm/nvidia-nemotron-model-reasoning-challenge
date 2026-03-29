from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Support direct execution from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.synthetic_public_like.common import ALICE_HEADER, format_public_decimal


def make_unit_conversion_puzzle(rng: random.Random, example_index: int) -> dict[str, Any]:
    """Generate a linear measurement conversion puzzle close to the public format."""
    multiplier = round(rng.uniform(0.50, 2.00), 4)
    example_count = rng.randint(3, 5)

    example_values = [round(rng.uniform(1.0, 50.0), 2) for _ in range(example_count)]
    query_value = round(rng.uniform(1.0, 50.0), 2)

    example_lines = "\n".join(
        f"{format_public_decimal(value, rng)} m becomes "
        f"{format_public_decimal(round(value * multiplier, 2), rng)}"
        for value in example_values
    )
    answer = format_public_decimal(round(query_value * multiplier, 2), rng)
    prompt = (
        f"{ALICE_HEADER}a secret unit conversion is applied to measurements. For example:\n"
        f"{example_lines}\n"
        f"Now, convert the following measurement: {format_public_decimal(query_value, rng)} m"
    )
    return {
        "id": f"syn_public_unit_conversion_{example_index:06d}",
        "prompt": prompt,
        "answer": answer,
        "category": "unit_conversion",
        "source": "synthetic_public_like_v1",
        "generator_family": "linear_scale_conversion",
        "hidden_multiplier": multiplier,
    }


if __name__ == "__main__":
    print(json.dumps(make_unit_conversion_puzzle(random.Random(42), 0), ensure_ascii=False, indent=2))
