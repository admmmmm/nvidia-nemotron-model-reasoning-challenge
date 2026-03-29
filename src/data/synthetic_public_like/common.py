from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


ALICE_HEADER = "In Alice's Wonderland, "
SYMBOL_CHARS = list("!\"#$%&'()*+-/:;<=>?@[\\]^`{|}")


def format_public_decimal(value: float, rng: random.Random | None = None) -> str:
    """Format decimals with the loose style used in the public prompts."""
    text = f"{value:.2f}"
    if text.endswith("00"):
        if rng is None or rng.random() < 0.85:
            return text[:-1]
        return text
    if text.endswith("0") and rng is not None and rng.random() < 0.5:
        return text[:-1]
    return text


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as UTF-8 JSONL for later SFT preprocessing."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
