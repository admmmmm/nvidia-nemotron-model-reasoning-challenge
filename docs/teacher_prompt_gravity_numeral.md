# Gravity And Numeral Teacher Prompt Notes

## Goal

Use an external teacher model to generate assistant targets for official public training
questions in categories where the base model often reasons correctly but loses points
because the final answer is formatted in a metric-hostile way.

This note currently covers:

- `gravity`
- `numeral`

`unit_conversion` is intentionally not covered here yet, because it does not appear to
share exactly the same dominant failure mode.

## Why These Two Categories Matter

### Gravity

For many `gravity` problems, the model appears to infer the hidden gravity constant
correctly and compute a numerically correct answer, but the final extracted answer is
wrong because the generated boxed answer contains units or extra LaTeX.

Typical bad endings:

```latex
\boxed{91.49 \text{ m}}
\boxed{32.0\text{ m}}
91.49\text{ m}
```

Why this fails:

- The metric first tries to extract the content inside `\boxed{}`.
- It then tries to parse the extracted answer as a float for numeric questions.
- Strings like `91.49\text{ m}` do not parse as floats, so they are judged as strings
  instead of numbers and become wrong.

The correct final answer format for `gravity` is:

```latex
\boxed{91.49}
```

Reasoning may mention units, but the boxed answer must contain only the bare number.

### Numeral

For many `numeral` problems, the model identifies the numeral system correctly, but the
final extracted answer is wrong because it nests LaTeX inside the box.

Typical bad ending:

```latex
\boxed{\text{XCII}}
```

Why this fails:

- The metric uses a simple regex to extract boxed content.
- Nested braces inside `\boxed{...}` break the extraction.
- `\boxed{\text{XCII}}` is extracted as something like `\text{XCII`, which is wrong.

The correct final answer format for `numeral` is:

```latex
\boxed{XCII}
```

No `\text{}` should appear inside the box.

## Training Target Strategy

These two categories do not need the same style as symbolic or cipher tasks.

Recommended target style:

- Keep reasoning concise but valid.
- End with a clean final answer block.
- Never put units or nested LaTeX inside the boxed answer.

Recommended final structure:

```text
<reasoning>

Final answer:
\boxed{...}
```

## Category-Specific Rules

### Gravity rules

- The reasoning may use units such as `m`, `s`, or `m/s^2`.
- The final boxed answer must be a pure numeric string.
- Do not put units inside the box.
- Do not append explanatory text after the boxed answer.
- Prefer a decimal answer close to the official answer.

Good:

```latex
\boxed{176.81}
```

Bad:

```latex
\boxed{176.81 \text{ m}}
\boxed{176.81m}
\boxed{about 176.81}
```

### Numeral rules

- The final boxed answer must be the raw numeral string.
- Do not wrap the numeral in `\text{}`.
- Do not add prose inside the box.
- Do not add punctuation inside the box.

Good:

```latex
\boxed{XCII}
```

Bad:

```latex
\boxed{\text{XCII}}
\boxed{Roman numeral XCII}
\boxed{XCII.}
```

## Recommended API Output Shape

For data generation, ask the teacher model to return JSON only:

```json
{
  "reasoning": "short but sufficient explanation",
  "boxed_answer": "\\boxed{...}",
  "assistant": "combined assistant target"
}
```

Where:

- `reasoning` is the human-readable explanation
- `boxed_answer` is the exact final answer string
- `assistant` is the full training target we will feed into SFT

## Validation Rules Before Keeping A Sample

### Gravity validation

- `boxed_answer` must match `^\\\\boxed\\{[-]?[0-9]+(?:\\.[0-9]+)?\\}$`
- no units
- no nested braces
- extracted numeric value must match the official answer within the metric tolerance

### Numeral validation

- `boxed_answer` must match `^\\\\boxed\\{[A-Z]+\\}$`
- no `\text{`
- no nested braces
- extracted answer must exactly match the official answer ignoring case

## Usage Pattern

The safest workflow is:

1. Use the official public prompt as input.
2. Provide the official known answer.
3. Ask the teacher model to explain the solution while preserving that answer.
4. Reject outputs whose `boxed_answer` violates the category rules.

This keeps the teacher focused on producing a high-quality assistant target rather than
guessing the answer from scratch.
