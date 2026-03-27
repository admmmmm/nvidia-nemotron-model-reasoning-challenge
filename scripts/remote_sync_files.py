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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    files_to_upload = [
        repo_root / ".env.server.example",
        repo_root / "scripts" / "run_nemotron_lora.sh",
        repo_root / "scripts" / "start_train_background.sh",
        repo_root / "scripts" / "sync_loop.sh",
        repo_root / "scripts" / "sync_run_to_git.sh",
        repo_root / "scripts" / "remote_push_and_start_train.py",
        repo_root / "scripts" / "remote_start_train.py",
        repo_root / "scripts" / "remote_stop_train.py",
        repo_root / "scripts" / "remote_tail_run.py",
        repo_root / "src" / "train" / "sft_local.py",
        repo_root / "src" / "train" / "lora_utils.py",
    ]

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
        if local_file.name == ".env.server.example":
            remote_dir = "/root/nemotron"
        elif local_file.parent.name == "scripts":
            remote_dir = "/root/nemotron/scripts"
        else:
            remote_dir = "/root/nemotron/src/train"
        remote_file = posixpath.join(remote_dir, local_file.name)
        upload_file(sftp, local_file, remote_file)
    sftp.close()

    remote_script = textwrap.dedent(
        """
        set -euo pipefail
        cd /root/nemotron
        echo "=== synced files ==="
        ls -l scripts/run_nemotron_lora.sh scripts/start_train_background.sh scripts/sync_loop.sh scripts/sync_run_to_git.sh || true
        ls -l scripts/remote_push_and_start_train.py scripts/remote_start_train.py scripts/remote_stop_train.py scripts/remote_tail_run.py || true
        ls -l src/train/sft_local.py src/train/lora_utils.py .env.server.example || true
        echo "=== verify key config ==="
        grep -n "disable-eval" src/train/sft_local.py || true
        grep -n "Patched Nemotron MoE dtype handlers" src/train/sft_local.py || true
        grep -n "NEMOTRON_TARGET_MODULES" src/train/lora_utils.py || true
        grep -n "LEARNING_RATE=1e-5" .env.server.example || true
        grep -n "DISABLE_EVAL=1" .env.server.example || true
        """
    ).strip()

    remote_path = "/root/nemotron/.tmp_remote_sync_files.sh"
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
