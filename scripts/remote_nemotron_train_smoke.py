from __future__ import annotations

import argparse
import sys
import textwrap

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated LoRA target modules, e.g. linear_qkv,linear_proj",
    )
    parser.add_argument(
        "--do-backward",
        action="store_true",
        help="Also run loss.backward() after the forward loss computation.",
    )
    args = parser.parse_args()

    target_modules = [item.strip() for item in args.targets.split(",") if item.strip()]
    target_literal = repr(target_modules)
    backward_snippet = "loss.backward()\nprint('backward ok', flush=True)" if args.do_backward else ""

    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd /root/nemotron
        export PATH=/home/vipuser/anaconda3/bin:$PATH
        source /home/vipuser/anaconda3/etc/profile.d/conda.sh
        conda activate /home/vipuser/anaconda3/envs/nemotron_mamba
        export PYTHONPATH=/root/nemotron:${{PYTHONPATH:-}}
        set -a
        source .env.server
        set +a

        python - <<'PY'
        import os
        import torch
        from peft import LoraConfig, get_peft_model

        from src.train.sft_local import (
            DataCollatorForCausalLM,
            build_model,
            build_tokenizer,
            encode_example,
            load_split_records,
        )

        class Args:
            model_name = os.environ["MODEL_NAME"]
            load_in_4bit = True
            low_cpu_mem_usage = True
            offload_state_dict = True
            offload_folder = "/root/nemotron/outputs/offload/train_smoke"
            max_memory_gpu = os.environ.get("MAX_MEMORY_GPU", "38GiB")
            max_memory_cpu = os.environ.get("MAX_MEMORY_CPU", "56GiB")

        target_modules = {target_literal}
        print("target_modules=", target_modules, flush=True)

        tokenizer = build_tokenizer(Args.model_name)
        train_file = "/root/nemotron/data/splits/default/train.jsonl"
        records = load_split_records(train_file)
        if not records:
            raise SystemExit(f"no train records found at {{train_file}}")

        example = encode_example(
            record=records[0],
            tokenizer=tokenizer,
            max_length=256,
            system_prompt=os.environ.get("SYSTEM_PROMPT", "You are a careful reasoning assistant."),
        )
        collator = DataCollatorForCausalLM(tokenizer.pad_token_id)
        batch = collator(
            [{{
                "input_ids": example.input_ids,
                "attention_mask": example.attention_mask,
                "labels": example.labels,
            }}]
        )

        model = build_model(Args)
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
        model.train()

        batch = {{k: v.to(model.device) for k, v in batch.items()}}
        print("input_ids.shape=", tuple(batch["input_ids"].shape), flush=True)
        print("labels.shape=", tuple(batch["labels"].shape), flush=True)

        outputs = model(**batch)
        loss = outputs.loss
        print("loss=", float(loss.detach().cpu()), flush=True)
        {backward_snippet}
        print("train smoke ok", flush=True)
        PY
        """
    ).strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
    )

    remote_path = "/root/nemotron/.tmp_remote_nemotron_train_smoke.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(remote_script)
    sftp.chmod(remote_path, 0o700)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_path}", timeout=3600)
    sys.stdout.write(stdout.read().decode("utf-8", "ignore"))
    sys.stderr.write(stderr.read().decode("utf-8", "ignore"))
    status = stdout.channel.recv_exit_status()
    client.exec_command(f"rm -f {remote_path}", timeout=30)
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
