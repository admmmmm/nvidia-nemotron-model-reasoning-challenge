from __future__ import annotations

import argparse
import sys

import paramiko


def build_python_script(
    target_modules: list[str],
    skip_modules: list[str],
    load_in_4bit: bool,
    do_backward: bool,
    max_length: int,
    min_valid_labels: int,
    sample_search_limit: int,
    max_memory_gpu: str,
    max_memory_cpu: str,
    local_files_only: bool,
) -> str:
    lines: list[str] = [
        "import os",
        "import torch",
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
        "from src.train.sft_local import DataCollatorForCausalLM, encode_example, load_split_records",
        "from src.data.format_sft import DEFAULT_SYSTEM_PROMPT",
        "",
        f"TARGET_MODULES = {target_modules!r}",
        f"SKIP_MODULES = {skip_modules!r}",
        f"LOAD_IN_4BIT = {load_in_4bit!r}",
        f"MAX_LENGTH = {max_length!r}",
        f"MIN_VALID_LABELS = {min_valid_labels!r}",
        f"SAMPLE_SEARCH_LIMIT = {sample_search_limit!r}",
        f"MAX_MEMORY_GPU = {max_memory_gpu!r}",
        f"MAX_MEMORY_CPU = {max_memory_cpu!r}",
        f"LOCAL_FILES_ONLY = {local_files_only!r}",
        "",
        "print('target_modules=', TARGET_MODULES, flush=True)",
        "print('skip_modules=', SKIP_MODULES, flush=True)",
        "print('load_in_4bit=', LOAD_IN_4BIT, flush=True)",
        "print('max_length=', MAX_LENGTH, flush=True)",
        "print('min_valid_labels=', MIN_VALID_LABELS, flush=True)",
        "print('sample_search_limit=', SAMPLE_SEARCH_LIMIT, flush=True)",
        "print('max_memory_gpu=', MAX_MEMORY_GPU, flush=True)",
        "print('max_memory_cpu=', MAX_MEMORY_CPU, flush=True)",
        "print('local_files_only=', LOCAL_FILES_ONLY, flush=True)",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_NAME'], trust_remote_code=True, use_fast=True)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "tokenizer.padding_side = 'right'",
        "",
        "records = load_split_records('/root/nemotron/data/splits/default/train.jsonl')",
        "if not records:",
        "    raise SystemExit('no train records found')",
        "",
        "collator = DataCollatorForCausalLM(tokenizer.pad_token_id)",
        "selected_index = None",
        "example = None",
        "batch = None",
        "valid_labels = -1",
        "search_limit = min(len(records), SAMPLE_SEARCH_LIMIT)",
        "for idx in range(search_limit):",
        "    candidate = encode_example(",
        "        record=records[idx],",
        "        tokenizer=tokenizer,",
        "        max_length=MAX_LENGTH,",
        "        system_prompt=os.environ.get('SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT),",
        "    )",
        "    candidate_batch = collator([{'input_ids': candidate.input_ids, 'attention_mask': candidate.attention_mask, 'labels': candidate.labels}])",
        "    candidate_valid = int((candidate_batch['labels'] != -100).sum().item())",
        "    if candidate_valid >= MIN_VALID_LABELS:",
        "        selected_index = idx",
        "        example = candidate",
        "        batch = candidate_batch",
        "        valid_labels = candidate_valid",
        "        break",
        "if batch is None:",
        "    raise SystemExit(f'no sample found with valid_labels >= {MIN_VALID_LABELS} in first {search_limit} records')",
        "print('selected_index=', selected_index, flush=True)",
        "print('valid_labels=', valid_labels, flush=True)",
        "",
        "model_kwargs = {",
        "    'trust_remote_code': True,",
        "    'device_map': 'auto',",
        "    'low_cpu_mem_usage': True,",
        "    'offload_state_dict': True,",
        "    'offload_folder': '/root/nemotron/outputs/offload/train_smoke',",
        "    'max_memory': {0: MAX_MEMORY_GPU, 'cpu': MAX_MEMORY_CPU},",
        "    'local_files_only': LOCAL_FILES_ONLY,",
        "}",
        "",
        "if LOAD_IN_4BIT:",
        "    bnb_kwargs = {",
        "        'load_in_4bit': True,",
        "        'bnb_4bit_quant_type': 'nf4',",
        "        'bnb_4bit_compute_dtype': torch.float16,",
        "        'bnb_4bit_use_double_quant': True,",
        "    }",
        "    if SKIP_MODULES:",
        "        bnb_kwargs['llm_int8_skip_modules'] = SKIP_MODULES",
        "    model_kwargs['quantization_config'] = BitsAndBytesConfig(**bnb_kwargs)",
        "else:",
        "    model_kwargs['torch_dtype'] = torch.float16",
        "",
        "model = AutoModelForCausalLM.from_pretrained(os.environ['MODEL_NAME'], **model_kwargs)",
        "if LOAD_IN_4BIT:",
        "    model = prepare_model_for_kbit_training(model)",
        "    if hasattr(model, 'enable_input_require_grads'):",
        "        model.enable_input_require_grads()",
        "    else:",
        "        def _make_inputs_require_grad(module, inputs, output):",
        "            output.requires_grad_(True)",
        "        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)",
        "model.config.use_cache = False",
        "",
    ]

    if target_modules:
        lines.extend(
            [
                "print('wrapping with LoRA...', flush=True)",
                "model = get_peft_model(",
                "    model,",
                "    LoraConfig(",
                "        r=16,",
                "        lora_alpha=32,",
                "        lora_dropout=0.05,",
                "        bias='none',",
                "        task_type='CAUSAL_LM',",
                "        target_modules=TARGET_MODULES,",
                "    ),",
                ")",
            ]
        )
    else:
        lines.append("print('running without LoRA wrapper...', flush=True)")

    lines.extend(
        [
            "model.train()",
            "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
            "print('trainable_params=', trainable_params, flush=True)",
            "batch = {k: v.to(model.device) for k, v in batch.items()}",
            "print('input_ids.shape=', tuple(batch['input_ids'].shape), flush=True)",
            "print('labels.shape=', tuple(batch['labels'].shape), flush=True)",
            "outputs = model(**batch)",
            "loss = outputs.loss",
            "print('loss=', float(loss.detach().cpu()), flush=True)",
        ]
    )

    if do_backward:
        lines.extend(["loss.backward()", "print('backward ok', flush=True)"])

    lines.append("print('train smoke ok', flush=True)")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--targets", default="", help="Comma-separated LoRA target modules.")
    parser.add_argument("--skip-modules", default="", help="Comma-separated quantization skip modules.")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-backward", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-valid-labels", type=int, default=1)
    parser.add_argument("--sample-search-limit", type=int, default=64)
    parser.add_argument("--max-memory-gpu", default="38GiB")
    parser.add_argument("--max-memory-cpu", default="32GiB")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    target_modules = [item.strip() for item in args.targets.split(",") if item.strip()]
    skip_modules = [item.strip() for item in args.skip_modules.split(",") if item.strip()]
    python_script = build_python_script(
        target_modules=target_modules,
        skip_modules=skip_modules,
        load_in_4bit=args.load_in_4bit,
        do_backward=args.do_backward,
        max_length=args.max_length,
        min_valid_labels=args.min_valid_labels,
        sample_search_limit=args.sample_search_limit,
        max_memory_gpu=args.max_memory_gpu,
        max_memory_cpu=args.max_memory_cpu,
        local_files_only=args.local_files_only,
    )
    shell_script = "\n".join(
        [
            "set -euo pipefail",
            "cd /root/nemotron",
            "export PATH=/home/vipuser/anaconda3/bin:$PATH",
            "source /home/vipuser/anaconda3/etc/profile.d/conda.sh",
            "conda activate /home/vipuser/anaconda3/envs/nemotron_mamba",
            "export PYTHONPATH=/root/nemotron:${PYTHONPATH:-}",
            "set -a",
            "source .env.server",
            "set +a",
            "python /root/nemotron/.tmp_remote_nemotron_train_smoke.py",
            "",
        ]
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
    )

    remote_py = "/root/nemotron/.tmp_remote_nemotron_train_smoke.py"
    remote_sh = "/root/nemotron/.tmp_remote_nemotron_train_smoke.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_py, "w") as handle:
        handle.write(python_script)
    with sftp.file(remote_sh, "w") as handle:
        handle.write(shell_script)
    sftp.chmod(remote_sh, 0o700)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_sh}", timeout=3600)
    sys.stdout.write(stdout.read().decode("utf-8", "ignore"))
    sys.stderr.write(stderr.read().decode("utf-8", "ignore"))
    status = stdout.channel.recv_exit_status()
    client.exec_command(f"rm -f {remote_py} {remote_sh}", timeout=30)
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
