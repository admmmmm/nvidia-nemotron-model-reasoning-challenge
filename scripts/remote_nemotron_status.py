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
    parser.add_argument("--pid", default="")
    args = parser.parse_args()

    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
        if [ -n "{args.pid}" ]; then
          echo "=== PROCESS ==="
          ps -fp {args.pid} || true
        fi
        echo "=== LOG FILES ==="
        cd /root/nemotron
        ls -1t outputs/logs/nemotron_lora_v0_*.log 2>/dev/null | head -n 6 || true
        latest_launcher="$(ls -1t outputs/logs/nemotron_lora_v0_launcher_*.log 2>/dev/null | head -n 1)"
        latest_train=""
        if [[ -f outputs/logs/nemotron_lora_v0_latest_log_path.txt ]]; then
          latest_train="$(cat outputs/logs/nemotron_lora_v0_latest_log_path.txt 2>/dev/null || true)"
        fi
        if [[ -z "$latest_train" ]]; then
          latest_train="$(find outputs/logs -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{{print $2}}')"
        fi
        if [[ -z "$latest_train" ]]; then
          latest_train="$(ls -1t outputs/logs/nemotron_lora_v0_*.log 2>/dev/null | grep -v launcher | head -n 1)"
        fi
        echo "=== LATEST LAUNCHER ==="
        echo "$latest_launcher"
        if [[ -n "$latest_launcher" ]]; then tail -n 80 "$latest_launcher" || true; fi
        echo "=== LATEST TRAIN ==="
        echo "$latest_train"
        if [[ -n "$latest_train" ]]; then tail -n 80 "$latest_train" || true; fi
        echo "=== STATUS ==="
        cat outputs/logs/nemotron_lora_v0_status.json 2>/dev/null || true
        echo "=== GPU ==="
        nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
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

    remote_path = "/root/nemotron/.tmp_remote_nemotron_status.sh"
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
