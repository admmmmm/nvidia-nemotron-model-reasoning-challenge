from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    # Support direct execution from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.synthetic_public_like.common import ALICE_HEADER


def _rotate_left(value: int, amount: int) -> int:
    amount %= 8
    return ((value << amount) | (value >> (8 - amount))) & 0xFF


def _rotate_right(value: int, amount: int) -> int:
    amount %= 8
    return ((value >> amount) | (value << (8 - amount))) & 0xFF


def _reverse_bits(value: int) -> int:
    return int(f"{value:08b}"[::-1], 2)


def _swap_nibbles(value: int) -> int:
    return ((value & 0x0F) << 4) | ((value & 0xF0) >> 4)


@dataclass(frozen=True)
class BitRule:
    name: str
    apply: Callable[[int], int]


def _build_rule_pool(rng: random.Random) -> list[BitRule]:
    """Sample a parameterized bit-operation pool so each puzzle has a hidden rule chain."""
    xor_mask = rng.randint(1, 255)
    and_mask = rng.randint(1, 255)
    or_mask = rng.randint(1, 255)
    left_rotate = rng.randint(1, 7)
    right_rotate = rng.randint(1, 7)
    left_shift = rng.randint(1, 3)
    right_shift = rng.randint(1, 3)

    return [
        BitRule(f"XOR 0x{xor_mask:02X}", lambda value, mask=xor_mask: value ^ mask),
        BitRule(f"AND 0x{and_mask:02X}", lambda value, mask=and_mask: value & mask),
        BitRule(f"OR 0x{or_mask:02X}", lambda value, mask=or_mask: value | mask),
        BitRule("NOT", lambda value: (~value) & 0xFF),
        BitRule(f"ROTATE LEFT {left_rotate}", lambda value, n=left_rotate: _rotate_left(value, n)),
        BitRule(f"ROTATE RIGHT {right_rotate}", lambda value, n=right_rotate: _rotate_right(value, n)),
        BitRule(f"SHIFT LEFT {left_shift}", lambda value, n=left_shift: (value << n) & 0xFF),
        BitRule(f"SHIFT RIGHT {right_shift}", lambda value, n=right_shift: value >> n),
        BitRule("REVERSE BITS", _reverse_bits),
        BitRule("SWAP NIBBLES", _swap_nibbles),
    ]


def make_bit_manipulation_puzzle(rng: random.Random, example_index: int) -> dict[str, Any]:
    """Generate an 8-bit transformation puzzle with a short hidden op chain."""
    chosen_rules = rng.sample(_build_rule_pool(rng), rng.randint(2, 4))

    def transform(value: int) -> int:
        for rule in chosen_rules:
            value = rule.apply(value) & 0xFF
        return value

    example_count = rng.randint(7, 10)
    seen_inputs: set[int] = set()
    examples: list[tuple[int, int]] = []
    while len(examples) < example_count:
        candidate = rng.randint(0, 255)
        if candidate in seen_inputs:
            continue
        seen_inputs.add(candidate)
        examples.append((candidate, transform(candidate)))

    query_value = rng.randint(0, 255)
    while query_value in seen_inputs:
        query_value = rng.randint(0, 255)

    example_lines = "\n".join(
        f"{source:08b} -> {target:08b}" for source, target in examples
    )
    prompt = (
        f"{ALICE_HEADER}a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, "
        "NOT, and possibly majority or choice functions.\n\n"
        f"Here are some examples of input -> output:\n{example_lines}\n\n"
        f"Now, determine the output for: {query_value:08b}"
    )
    return {
        "id": f"syn_public_bit_manipulation_{example_index:06d}",
        "prompt": prompt,
        "answer": f"{transform(query_value):08b}",
        "category": "bit_manipulation",
        "source": "synthetic_public_like_v1",
        "generator_family": "bit_op_chain",
        "hidden_rules": [rule.name for rule in chosen_rules],
    }


if __name__ == "__main__":
    print(json.dumps(make_bit_manipulation_puzzle(random.Random(42), 0), ensure_ascii=False, indent=2))
