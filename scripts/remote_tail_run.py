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
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd /root/nemotron
        echo "=== process check ==="
        ps -ef | grep -E "{args.run_name}|run_nemotron_lora|src.train.sft_local|sync_loop.sh" | grep -v grep || true
        echo "=== pointers ==="
        cat outputs/logs/{args.run_name}_latest_log_dir.txt 2>/dev/null || true
        cat outputs/logs/{args.run_name}_latest_log_path.txt 2>/dev/null || true
        run_log_path=""
        if [[ -f outputs/logs/{args.run_name}_latest_log_path.txt ]]; then
          run_log_path="$(cat outputs/logs/{args.run_name}_latest_log_path.txt)"
        fi
        echo "=== launch latest ==="
        tail -n 80 outputs/logs/launch_latest.out 2>/dev/null || true
        echo "=== run log ==="
        if [[ -n "$run_log_path" && -f "$run_log_path" ]]; then
          tail -n 120 "$run_log_path"
        else
          echo "missing run log"
        fi
        echo "=== gpu ==="
        nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
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

    remote_path = "/root/nemotron/.tmp_remote_tail_run.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(remote_script)
    sftp.chmod(remote_path, 0o700)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_path}", timeout=300)
    sys.stdout.write(stdout.read().decode("utf-8", "ignore"))
    sys.stderr.write(stderr.read().decode("utf-8", "ignore"))
    status = stdout.channel.recv_exit_status()
    client.exec_command(f"rm -f {remote_path}", timeout=30)
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
