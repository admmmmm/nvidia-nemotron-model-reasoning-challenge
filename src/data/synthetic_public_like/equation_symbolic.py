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

from src.data.synthetic_public_like.common import ALICE_HEADER, SYMBOL_CHARS


OPERATOR_POOL = list("*-+)!'{}#`|<>&@!?")


@dataclass(frozen=True)
class SymbolRule:
    name: str
    apply: Callable[[str, dict[str, str]], str]


def _collapse_consecutive(text: str) -> str:
    if not text:
        return text
    pieces = [text[0]]
    for char in text[1:]:
        if char != pieces[-1]:
            pieces.append(char)
    return "".join(pieces)


def _unique_in_order(text: str) -> str:
    pieces: list[str] = []
    for char in text:
        if char not in pieces:
            pieces.append(char)
    return "".join(pieces)


def _mapped(chars: str, char_map: dict[str, str]) -> str:
    return "".join(char_map[char] for char in chars)


SYMBOL_RULES = [
    SymbolRule("drop_middle_keep_outer", lambda text, char_map: text[0] + text[1] + text[3] + text[4]),
    SymbolRule("keep_edges", lambda text, char_map: text[0] + text[4]),
    SymbolRule("keep_inner_pair", lambda text, char_map: text[1] + text[3]),
    SymbolRule("reverse_outer", lambda text, char_map: text[4] + text[3] + text[1] + text[0]),
    SymbolRule("mapped_outer", lambda text, char_map: _mapped(text[0] + text[1] + text[3] + text[4], char_map)),
    SymbolRule("mapped_reverse_outer", lambda text, char_map: _mapped(text[4] + text[3] + text[1] + text[0], char_map)),
    SymbolRule("mapped_dedup_outer", lambda text, char_map: _collapse_consecutive(_mapped(text[0] + text[1] + text[3] + text[4], char_map))),
    SymbolRule("mapped_unique_outer", lambda text, char_map: _unique_in_order(_mapped(text[0] + text[1] + text[3] + text[4], char_map))),
    SymbolRule("mapped_keep_edges", lambda text, char_map: _mapped(text[0] + text[4], char_map)),
    SymbolRule("mapped_inner_triplet", lambda text, char_map: _mapped(text[1] + text[3] + text[4], char_map)),
]


def _build_char_map(rng: random.Random) -> dict[str, str]:
    shuffled = SYMBOL_CHARS[:]
    rng.shuffle(shuffled)
    return dict(zip(SYMBOL_CHARS, shuffled))


def _make_symbol_input(rng: random.Random, operator: str, outer_pool: list[str]) -> str:
    return (
        rng.choice(outer_pool)
        + rng.choice(outer_pool)
        + operator
        + rng.choice(outer_pool)
        + rng.choice(outer_pool)
    )


def make_equation_symbolic_puzzle(rng: random.Random, example_index: int) -> dict[str, Any]:
    """Generate a symbolic operator puzzle with center-operator transformations."""
    operator_count = rng.randint(2, 3)
    chosen_operators = rng.sample(OPERATOR_POOL, operator_count)
    chosen_rules = rng.sample(SYMBOL_RULES, operator_count)
    operator_to_rule = dict(zip(chosen_operators, chosen_rules))
    char_map = _build_char_map(rng)

    outer_pool = [char for char in SYMBOL_CHARS if char not in chosen_operators]
    if len(outer_pool) < 8:
        outer_pool = SYMBOL_CHARS[:]

    example_count = rng.randint(3, 5)
    operator_schedule = chosen_operators[:]
    while len(operator_schedule) < example_count:
        operator_schedule.append(rng.choice(chosen_operators))
    rng.shuffle(operator_schedule)

    examples: list[tuple[str, str]] = []
    for operator in operator_schedule:
        candidate = _make_symbol_input(rng, operator, outer_pool)
        answer = operator_to_rule[operator].apply(candidate, char_map)
        examples.append((candidate, answer))

    query_operator = rng.choice(chosen_operators)
    query = _make_symbol_input(rng, query_operator, outer_pool)
    answer = operator_to_rule[query_operator].apply(query, char_map)

    example_lines = "\n".join(f"{left} = {right}" for left, right in examples)
    prompt = (
        f"{ALICE_HEADER}a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        f"{example_lines}\n"
        f"Now, determine the result for: {query}"
    )
    return {
        "id": f"syn_public_equation_symbolic_{example_index:06d}",
        "prompt": prompt,
        "answer": answer,
        "category": "equation_symbolic",
        "source": "synthetic_public_like_v1",
        "generator_family": "center_operator_symbol_transform",
        "hidden_rules": {operator: rule.name for operator, rule in operator_to_rule.items()},
    }


if __name__ == "__main__":
    print(json.dumps(make_equation_symbolic_puzzle(random.Random(42), 0), ensure_ascii=False, indent=2))
