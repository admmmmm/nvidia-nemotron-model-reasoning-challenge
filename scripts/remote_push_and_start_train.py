from __future__ import annotations

import argparse
import posixpath
import sys
import textwrap
from pathlib import Path

import paramiko


def upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    sftp.put(str(local_path), remote_path)
    sftp.chmod(remote_path, 0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-folder-name", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    files_to_upload = [
        repo_root / "scripts" / "run_nemotron_lora.sh",
        repo_root / "scripts" / "start_train_background.sh",
        repo_root / "scripts" / "sync_loop.sh",
        repo_root / "scripts" / "sync_run_to_git.sh",
        repo_root / "src" / "train" / "sft_local.py",
        repo_root / "src" / "train" / "lora_utils.py",
    ]

    run_folder_name = args.run_folder_name or args.run_name

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
    )

    sftp = client.open_sftp()
    for local_file in files_to_upload:
        if local_file.parent.name == "scripts":
            remote_dir = "/root/nemotron/scripts"
        else:
            remote_dir = "/root/nemotron/src/train"
        remote_file = posixpath.join(remote_dir, local_file.name)
        upload_file(sftp, local_file, remote_file)
    sftp.close()

    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd /root/nemotron
        pkill -f "sync_loop.sh /root/nemotron /root/nemotron_git {args.run_name}" >/dev/null 2>&1 || true
        pkill -f "sync_loop.sh /root/nemotron /root/nemotron_git nemotron_lora_v0" >/dev/null 2>&1 || true
        pkill -f "bash /root/nemotron/scripts/run_nemotron_lora.sh" >/dev/null 2>&1 || true
        pkill -f "python -m src.train.sft_local" >/dev/null 2>&1 || true
        sleep 2
        echo "=== verify synced source ==="
        grep -n "Patched Nemotron MoE dtype handlers" src/train/sft_local.py || true
        grep -n "autocast_adapter_dtype" src/train/sft_local.py || true
        grep -n 'NEMOTRON_TARGET_MODULES' src/train/lora_utils.py || true
        export RUN_NAME="{args.run_name}"
        export RUN_FOLDER_NAME="{run_folder_name}"
        export GIT_SYNC_SOURCE_REPO="/root/nemotron"
        export GIT_SYNC_REPO="/root/nemotron_git"
        export SYNC_INTERVAL_SECONDS="60"
        bash scripts/start_train_background.sh "{run_folder_name}"
        echo "=== process check ==="
        ps -ef | grep -E "run_nemotron_lora|src.train.sft_local|sync_loop.sh" | grep -v grep || true
        echo "=== launch tail ==="
        tail -n 80 outputs/logs/launch_latest.out 2>/dev/null || true
        """
    ).strip()

    remote_path = "/root/nemotron/.tmp_remote_push_and_start_train.sh"
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
