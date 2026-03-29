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


OPERATOR_POOL = list("+-*/%!?<>{}[]()@#$^'`\\|:&")


@dataclass(frozen=True)
class NumericRule:
    name: str
    apply: Callable[[int, int, int, int, str], str]


def _ab(a: int, b: int) -> int:
    return 10 * a + b


NUMERIC_RULES = [
    NumericRule("concat_forward", lambda a, b, c, d, op: f"{a}{b}{c}{d}"),
    NumericRule("concat_reversed_pairs", lambda a, b, c, d, op: f"{c}{d}{a}{b}"),
    NumericRule("sum_two_numbers", lambda a, b, c, d, op: str(_ab(a, b) + _ab(c, d))),
    NumericRule("difference_two_numbers", lambda a, b, c, d, op: str(_ab(a, b) - _ab(c, d))),
    NumericRule("absolute_difference", lambda a, b, c, d, op: str(abs(_ab(a, b) - _ab(c, d)))),
    NumericRule("digit_sums_concat", lambda a, b, c, d, op: f"{a + c}{b + d}"),
    NumericRule("digit_absdiff_concat", lambda a, b, c, d, op: f"{abs(a - c)}{abs(b - d)}"),
    NumericRule("cross_sums_concat", lambda a, b, c, d, op: f"{a + d}{b + c}"),
    NumericRule("padded_digit_sums", lambda a, b, c, d, op: f"{a + c:02d}{b + d:02d}"),
    NumericRule("padded_digit_products", lambda a, b, c, d, op: f"{a * c:02d}{b * d:02d}"),
    NumericRule("prefix_operator_absdiff", lambda a, b, c, d, op: f"{op}{abs(_ab(a, b) - _ab(c, d))}"),
    NumericRule("suffix_operator_absdiff", lambda a, b, c, d, op: f"{abs(_ab(a, b) - _ab(c, d))}{op}"),
]


def _format_equation(left_a: int, left_b: int, operator: str, right_a: int, right_b: int) -> str:
    return f"{left_a}{left_b}{operator}{right_a}{right_b}"


def make_equation_numeric_puzzle(rng: random.Random, example_index: int) -> dict[str, Any]:
    """Generate a small operator-overloading puzzle over two-digit numbers."""
    operator_count = rng.randint(2, 3)
    chosen_operators = rng.sample(OPERATOR_POOL, operator_count)
    chosen_rules = rng.sample(NUMERIC_RULES, operator_count)
    operator_to_rule = dict(zip(chosen_operators, chosen_rules))

    example_count = rng.randint(3, 5)
    operator_schedule = chosen_operators[:]
    while len(operator_schedule) < example_count:
        operator_schedule.append(rng.choice(chosen_operators))
    rng.shuffle(operator_schedule)

    examples: list[tuple[str, str]] = []
    for operator in operator_schedule:
        a, b, c, d = [rng.randint(0, 9) for _ in range(4)]
        rule = operator_to_rule[operator]
        examples.append((_format_equation(a, b, operator, c, d), rule.apply(a, b, c, d, operator)))

    query_operator = rng.choice(chosen_operators)
    qa, qb, qc, qd = [rng.randint(0, 9) for _ in range(4)]
    query = _format_equation(qa, qb, query_operator, qc, qd)
    answer = operator_to_rule[query_operator].apply(qa, qb, qc, qd, query_operator)

    example_lines = "\n".join(f"{left} = {right}" for left, right in examples)
    prompt = (
        f"{ALICE_HEADER}a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        f"{example_lines}\n"
        f"Now, determine the result for: {query}"
    )
    return {
        "id": f"syn_public_equation_numeric_{example_index:06d}",
        "prompt": prompt,
        "answer": answer,
        "category": "equation_numeric",
        "source": "synthetic_public_like_v1",
        "generator_family": "operator_overloading_numeric",
        "hidden_rules": {operator: rule.name for operator, rule in operator_to_rule.items()},
    }


if __name__ == "__main__":
    print(json.dumps(make_equation_numeric_puzzle(random.Random(42), 0), ensure_ascii=False, indent=2))
