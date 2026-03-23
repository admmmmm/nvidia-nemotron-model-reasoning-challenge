#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_OPENAI_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4-turbo"]
DEFAULT_OPENAI_ENCODINGS = ["o200k_base", "cl100k_base"]
DEFAULT_OPENAI_FALLBACK_ENCODING = "o200k_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokens for Hugging Face and OpenAI tokenizers."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Read prompt text from a UTF-8 file.",
    )
    parser.add_argument(
        "--text",
        help="Read prompt text directly from a command-line argument.",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_MODEL,
        help=f"Hugging Face model id. Default: {DEFAULT_HF_MODEL}",
    )
    parser.add_argument(
        "--openai-model",
        action="append",
        dest="openai_models",
        help=(
            "OpenAI model name for tiktoken counting. "
            "May be provided multiple times. "
            f"Defaults: {', '.join(DEFAULT_OPENAI_MODELS)}"
        ),
    )
    parser.add_argument(
        "--encoding",
        action="append",
        dest="encodings",
        help=(
            "Raw tiktoken encoding name to count. "
            "May be provided multiple times. "
            f"Defaults: {', '.join(DEFAULT_OPENAI_ENCODINGS)}"
        ),
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip Hugging Face chat-template token counting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of plain text.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Optional context window size. Report remaining tokens for each count.",
    )
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.file and args.text is not None:
        raise SystemExit("Use either --file or --text, not both.")

    if args.text is not None:
        return args.text

    if args.file:
        return args.file.read_text(encoding="utf-8")

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise SystemExit("Provide --file, --text, or pipe text through stdin.")


def count_hf_tokens(text: str, model_id: str, include_chat_template: bool) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'transformers'. Install it in the Python "
            "environment used to run this script."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    with_special_ids = tokenizer(text, add_special_tokens=True)["input_ids"]

    result: dict[str, Any] = {
        "model": model_id,
        "raw_tokens": len(raw_ids),
        "with_special_tokens": len(with_special_ids),
        "chat_template_tokens": None,
    }

    if include_chat_template and getattr(tokenizer, "chat_template", None):
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        result["chat_template_tokens"] = len(chat_ids)

    return result


def count_openai_model_tokens(
    text: str,
    model_name: str,
    fallback_encoding_name: str = DEFAULT_OPENAI_FALLBACK_ENCODING,
) -> dict[str, Any]:
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'tiktoken'. Install it in the Python "
            "environment used to run this script."
        ) from exc

    used_fallback = False

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding(fallback_encoding_name)
        used_fallback = True

    token_count = len(encoding.encode(text))
    result = {
        "model": model_name,
        "encoding": encoding.name,
        "tokens": token_count,
        "used_fallback_encoding": used_fallback,
    }
    if used_fallback:
        result["fallback_reason"] = (
            f"Local tiktoken could not map model '{model_name}' automatically."
        )
    return result


def count_openai_encoding_tokens(text: str, encoding_name: str) -> dict[str, Any]:
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'tiktoken'. Install it in the Python "
            "environment used to run this script."
        ) from exc

    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text))
    return {
        "encoding": encoding_name,
        "tokens": token_count,
    }


def build_report(args: argparse.Namespace, text: str) -> dict[str, Any]:
    openai_models = args.openai_models or DEFAULT_OPENAI_MODELS
    encodings = args.encodings or DEFAULT_OPENAI_ENCODINGS

    report: dict[str, Any] = {
        "text_stats": {
            "characters": len(text),
            "lines": len(text.splitlines()),
            "words": len(text.split()),
        },
        "huggingface": count_hf_tokens(
            text=text,
            model_id=args.hf_model,
            include_chat_template=not args.no_chat_template,
        ),
        "openai_models": [],
        "openai_encodings": [],
    }

    for model_name in openai_models:
        report["openai_models"].append(count_openai_model_tokens(text, model_name))

    for encoding_name in encodings:
        report["openai_encodings"].append(count_openai_encoding_tokens(text, encoding_name))

    if args.max_model_len is not None:
        max_len = args.max_model_len
        hf = report["huggingface"]
        hf["remaining_raw_tokens"] = max_len - hf["raw_tokens"]
        if hf["chat_template_tokens"] is not None:
            hf["remaining_chat_template_tokens"] = max_len - hf["chat_template_tokens"]

        for item in report["openai_models"]:
            item["remaining_tokens"] = max_len - item["tokens"]

        for item in report["openai_encodings"]:
            item["remaining_tokens"] = max_len - item["tokens"]

    return report


def print_plain(report: dict[str, Any]) -> None:
    stats = report["text_stats"]
    hf = report["huggingface"]

    print("Text")
    print(f"  characters: {stats['characters']}")
    print(f"  lines: {stats['lines']}")
    print(f"  words: {stats['words']}")

    print("Hugging Face")
    print(f"  model: {hf['model']}")
    print(f"  raw_tokens: {hf['raw_tokens']}")
    print(f"  with_special_tokens: {hf['with_special_tokens']}")
    if "remaining_raw_tokens" in hf:
        print(f"  remaining_raw_tokens: {hf['remaining_raw_tokens']}")
    if hf["chat_template_tokens"] is None:
        print("  chat_template_tokens: N/A")
    else:
        print(f"  chat_template_tokens: {hf['chat_template_tokens']}")
        if "remaining_chat_template_tokens" in hf:
            print(
                "  remaining_chat_template_tokens: "
                f"{hf['remaining_chat_template_tokens']}"
            )

    print("OpenAI Models")
    for item in report["openai_models"]:
        suffix = ""
        if item.get("used_fallback_encoding"):
            suffix = " [fallback]"
        print(
            f"  {item['model']}: {item['tokens']} tokens "
            f"(encoding={item['encoding']}){suffix}"
        )
        if "remaining_tokens" in item:
            print(f"    remaining_tokens: {item['remaining_tokens']}")

    print("OpenAI Encodings")
    for item in report["openai_encodings"]:
        print(f"  {item['encoding']}: {item['tokens']} tokens")
        if "remaining_tokens" in item:
            print(f"    remaining_tokens: {item['remaining_tokens']}")


def main() -> None:
    args = parse_args()
    text = load_text(args)
    report = build_report(args, text)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print_plain(report)


if __name__ == "__main__":
    main()
