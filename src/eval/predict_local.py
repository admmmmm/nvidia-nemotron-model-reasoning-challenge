from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.format_sft import DEFAULT_SYSTEM_PROMPT, load_jsonl_records


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference for validation or test generation.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-file", default="data/splits/default/val.jsonl")
    parser.add_argument("--output-file", default="outputs/logs/val_predictions.csv")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model with 4-bit quantization.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name: str, load_in_4bit: bool):
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model


def maybe_load_adapter(model, adapter_dir: str | None):
    if not adapter_dir:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return load_jsonl_records(input_path)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_prompt_tokens(tokenizer, prompt: str, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    return encoded


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args.model_name)
    model = build_model(args.model_name, args.load_in_4bit)
    model = maybe_load_adapter(model, args.adapter_dir)
    model.eval()

    rows = load_rows(args.input_file)
    predictions = []

    for row in rows:
        input_ids = build_prompt_tokens(tokenizer, str(row["prompt"]), args.system_prompt)
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append({"id": row["id"], "prediction": prediction})

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)

    config_path = output_path.with_suffix(".json")
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
