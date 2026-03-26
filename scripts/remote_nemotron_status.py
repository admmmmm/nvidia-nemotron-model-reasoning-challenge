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
        ls -1t outputs/logs/nemotron_lora_v0_*.log | head -n 6
        latest_launcher="$(ls -1t outputs/logs/nemotron_lora_v0_launcher_*.log | head -n 1)"
        latest_train="$(ls -1t outputs/logs/nemotron_lora_v0_*.log | grep -v launcher | head -n 1)"
        echo "=== LATEST LAUNCHER ==="
        echo "$latest_launcher"
        tail -n 80 "$latest_launcher" || true
        echo "=== LATEST TRAIN ==="
        echo "$latest_train"
        tail -n 80 "$latest_train" || true
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
