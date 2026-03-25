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
    args = parser.parse_args()

    remote_script = textwrap.dedent(
        r"""
        set -euo pipefail
        cd /root/nemotron
        export PATH=/home/vipuser/anaconda3/bin:$PATH
        source /home/vipuser/anaconda3/etc/profile.d/conda.sh
        conda activate /home/vipuser/anaconda3/envs/nemotron_mamba
        export PYTHONPATH=/root/nemotron:${PYTHONPATH:-}
        set -a
        source .env.server
        set +a

        python - <<'PY'
        import os
        import torch
        from transformers import AutoTokenizer
        from peft import get_peft_model

        from src.train.sft_local import build_model
        from src.train.lora_utils import build_lora_config, choose_target_modules

        class Args:
            model_name = os.environ["MODEL_NAME"]
            load_in_4bit = True
            low_cpu_mem_usage = True
            offload_state_dict = True
            offload_folder = "/root/nemotron/outputs/offload/smoke_lora"
            max_memory_gpu = os.environ.get("MAX_MEMORY_GPU", "38GiB")
            max_memory_cpu = os.environ.get("MAX_MEMORY_CPU", "56GiB")

        print("target_modules=", choose_target_modules(Args.model_name), flush=True)
        print("loading tokenizer...", flush=True)
        tok = AutoTokenizer.from_pretrained(Args.model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print("loading base model...", flush=True)
        model = build_model(Args)

        print("wrapping with LoRA...", flush=True)
        model = get_peft_model(
            model,
            build_lora_config(
                model_name=Args.model_name,
                rank=16,
                alpha=32,
                dropout=0.05,
            ),
        )

        text = "What is 2 + 2?"
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print("running forward...", flush=True)
        with torch.no_grad():
            out = model(**inputs)

        print("forward ok", tuple(out.logits.shape), flush=True)
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

    remote_path = "/root/nemotron/.tmp_remote_nemotron_smoke.sh"
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
