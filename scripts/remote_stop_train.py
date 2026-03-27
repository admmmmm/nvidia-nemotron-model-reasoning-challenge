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
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    run_grep = args.run_name if args.run_name else "run_nemotron_lora|src.train.sft_local|sync_loop.sh"
    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd /root/nemotron
        pkill -f "python -m src.train.sft_local" >/dev/null 2>&1 || true
        pkill -f "bash /root/nemotron/scripts/run_nemotron_lora.sh" >/dev/null 2>&1 || true
        if [[ -n "{args.run_name}" ]]; then
          pkill -f "tee outputs/logs/{args.run_name}/" >/dev/null 2>&1 || true
        fi
        if [[ -n "{args.run_name}" ]]; then
          pkill -f "sync_loop.sh /root/nemotron /root/nemotron_git {args.run_name}" >/dev/null 2>&1 || true
        fi
        sleep 2
        echo "=== remaining processes ==="
        ps -ef | grep -E "{run_grep}" | grep -v grep || true
        echo "=== latest pointers ==="
        cat outputs/logs/{args.run_name}_latest_log_dir.txt 2>/dev/null || true
        cat outputs/logs/{args.run_name}_latest_log_path.txt 2>/dev/null || true
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

    remote_path = "/root/nemotron/.tmp_remote_stop_train.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(remote_script)
    sftp.chmod(remote_path, 0o700)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_path}", timeout=120)
    sys.stdout.write(stdout.read().decode("utf-8", "ignore"))
    sys.stderr.write(stderr.read().decode("utf-8", "ignore"))
    status = stdout.channel.recv_exit_status()
    client.exec_command(f"rm -f {remote_path}", timeout=30)
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
