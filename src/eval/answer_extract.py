from __future__ import annotations

import re


NUMERIC_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:/\d+)?(?:[eE][-+]?\d+)?"
)


def extract_boxed_answer(text: str) -> str | None:
    marker = "\\boxed"
    start = 0
    matches: list[str] = []

    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break

        cursor = idx + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            start = idx + len(marker)
            continue

        cursor += 1
        depth = 1
        content: list[str] = []

        while cursor < len(text):
            char = text[cursor]
            next_char = text[cursor + 1] if cursor + 1 < len(text) else ""

            if char == "\\" and next_char in "{}":
                content.append(next_char)
                cursor += 2
                continue
            if char == "{":
                depth += 1
                content.append(char)
                cursor += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    matches.append("".join(content).strip())
                    break
                content.append(char)
                cursor += 1
                continue

            content.append(char)
            cursor += 1

        start = idx + len(marker)

    if not matches:
        return None
    return matches[-1]


def extract_last_numeric_answer(text: str) -> str | None:
    matches = NUMERIC_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    stripped = text.strip()
    if not stripped:
        return ""

    numeric = extract_last_numeric_answer(stripped)
    if numeric:
        return numeric

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]
