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
        export PATH=/home/vipuser/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
        cd /root/nemotron
        if [[ ! -f .env.server ]]; then
          echo ".env.server missing" >&2
          exit 1
        fi
        python3 - <<'PY'
from pathlib import Path

path = Path("/root/nemotron/.env.server")
text = path.read_text(encoding="utf-8")

def set_key(src: str, key: str, value: str) -> str:
    import re
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    repl = f"{key}={value}"
    if re.search(pattern, src):
        return re.sub(pattern, repl, src)
    if not src.endswith("\n"):
        src += "\n"
    return src + repl + "\n"

updates = {
    "OUTPUT_DIR": "outputs/adapters/${RUN_NAME}",
    "MAX_LENGTH": "1024",
    "LEARNING_RATE": "1e-5",
    "EVAL_STEPS": "500",
    "MAX_GRAD_NORM": "1.0",
    "EVALUATION_STRATEGY": "no",
    "SAVE_STRATEGY": "epoch",
    "DISABLE_EVAL": "1",
    "MAX_MEMORY_GPU": "39GiB",
    "MAX_MEMORY_CPU": "48GiB",
}

for key, value in updates.items():
    text = set_key(text, key, value)

path.write_text(text, encoding="utf-8")
PY
        echo "=== updated .env.server ==="
        grep -nE '^(OUTPUT_DIR|MAX_LENGTH|LEARNING_RATE|EVAL_STEPS|MAX_GRAD_NORM|EVALUATION_STRATEGY|SAVE_STRATEGY|DISABLE_EVAL|MAX_MEMORY_GPU|MAX_MEMORY_CPU)=' .env.server || true
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

    remote_path = "/root/nemotron/.tmp_remote_update_server_env.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(remote_script)
    sftp.chmod(remote_path, 0o700)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_path}", timeout=180)
    sys.stdout.write(stdout.read().decode("utf-8", "ignore"))
    sys.stderr.write(stderr.read().decode("utf-8", "ignore"))
    status = stdout.channel.recv_exit_status()
    client.exec_command(f"rm -f {remote_path}", timeout=30)
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
