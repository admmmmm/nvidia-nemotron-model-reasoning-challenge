from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.data.format_sft import DEFAULT_SYSTEM_PROMPT, build_assistant_target, load_jsonl_records
from src.train.lora_utils import build_lora_config


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NEMOTRON_4BIT_SKIP_MODULES = ["in_proj", "out_proj", "x_proj", "dt_proj"]


@dataclass
class SFTExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class DataCollatorForCausalLM:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local QLoRA SFT baseline.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-file", default="data/splits/default/train.jsonl")
    parser.add_argument("--val-file", default="data/splits/default/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/adapters/qwen_baseline")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model with 4-bit quantization.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when formatting each training row.",
    )
    parser.add_argument("--max-memory-gpu", default="39GiB")
    parser.add_argument("--max-memory-cpu", default="32GiB")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load tokenizer/model only from local HF cache.",
    )
    parser.add_argument(
        "--force-full-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force the full model onto GPU 0 and disable CPU offload.",
    )
    return parser.parse_args()


def load_split_records(path: str | Path) -> list[dict[str, Any]]:
    split_path = Path(path)
    if not split_path.exists():
        return []
    return load_jsonl_records(split_path)


def build_tokenizer(model_name: str, local_files_only: bool = False) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _normalize_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "ids"):
        return list(tokenized.ids)
    if isinstance(tokenized, dict):
        input_ids = tokenized["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        input_ids = tokenized["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)
    return list(tokenized)



def encode_example(
    record: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    system_prompt: str,
) -> SFTExample:
    answer_text = build_assistant_target(str(record["answer"]))
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(record["prompt"])},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": answer_text},
    ]

    prompt_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    full_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))

    labels = [-100] * prompt_length + full_ids[prompt_length:]
    attention_mask = [1] * len(full_ids)
    return SFTExample(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def build_dataset(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    system_prompt: str,
) -> Dataset:
    encoded_rows = []
    for record in records:
        example = encode_example(
            record=record,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
        )
        encoded_rows.append(
            {
                "input_ids": example.input_ids,
                "attention_mask": example.attention_mask,
                "labels": example.labels,
            }
        )
    return Dataset.from_list(encoded_rows)


def build_model(
    model_name: str,
    load_in_4bit: bool,
    max_memory_gpu: str,
    max_memory_cpu: str,
    local_files_only: bool,
    force_full_gpu: bool,
) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }

    if force_full_gpu:
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["max_memory"] = {0: max_memory_gpu}
    else:
        model_kwargs.update(
            {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "offload_state_dict": True,
                "offload_folder": "outputs/offload/train_main",
                "max_memory": {0: max_memory_gpu, "cpu": max_memory_cpu},
            }
        )

    if load_in_4bit:
        quant_kwargs: dict[str, Any] = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if "nemotron" in model_name.lower():
            # Empirically required on the HF remote-code path: Mamba kernel
            # projections cannot be passed through the generic 4-bit wrapper.
            quant_kwargs["llm_int8_skip_modules"] = NEMOTRON_4BIT_SKIP_MODULES
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    model.config.use_cache = False
    return model


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args.model_name, local_files_only=args.local_files_only)

    train_records = load_split_records(args.train_file)
    val_records = load_split_records(args.val_file)
    if args.train_limit is not None:
        train_records = train_records[: args.train_limit]
    if args.val_limit is not None:
        val_records = val_records[: args.val_limit]
    if not train_records:
        raise SystemExit(
            "Training split is missing. Run "
            "`python -m src.data.preprocess --train-file train.csv` first."
        )

    train_dataset = build_dataset(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )
    eval_dataset = (
        build_dataset(
            val_records,
            tokenizer=tokenizer,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
        )
        if val_records
        else None
    )

    model = build_model(
        args.model_name,
        args.load_in_4bit,
        max_memory_gpu=args.max_memory_gpu,
        max_memory_cpu=args.max_memory_cpu,
        local_files_only=args.local_files_only,
        force_full_gpu=args.force_full_gpu,
    )
    model = get_peft_model(
        model,
        build_lora_config(
            model_name=args.model_name,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        ),
    )
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad_(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )
    evaluation_value = "steps" if eval_dataset is not None else "no"
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_signature.parameters:
        training_kwargs["evaluation_strategy"] = evaluation_value
    else:
        training_kwargs["eval_strategy"] = evaluation_value

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
