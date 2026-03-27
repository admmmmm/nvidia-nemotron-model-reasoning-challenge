from __future__ import annotations

import argparse
import os
import posixpath
import stat
import subprocess
import sys
import time
from pathlib import Path

import paramiko


def download_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> bool:
    try:
        attr = sftp.stat(remote_path)
    except OSError:
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        stat = local_path.stat()
        if stat.st_size == attr.st_size and int(stat.st_mtime) >= int(attr.st_mtime):
            return True
    sftp.get(remote_path, str(local_path))
    os.utime(local_path, (attr.st_atime, attr.st_mtime))
    return True


def download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> bool:
    try:
        entries = sftp.listdir_attr(remote_dir)
    except OSError:
        return False

    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        remote_child = posixpath.join(remote_dir, entry.filename)
        local_child = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_dir(sftp, remote_child, local_child)
        else:
            download_file(sftp, remote_child, local_child)
    return True


def sync_once(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    local_logs = repo_root / "outputs" / "logs"
    local_logs.mkdir(parents=True, exist_ok=True)

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

    remote_logs = f"{args.remote_repo.rstrip('/')}/outputs/logs"
    base_files = [
        "launch_latest.out",
        "launch_latest.pid",
        f"{args.run_name}_latest_log_dir.txt",
        f"{args.run_name}_latest_log_path.txt",
        f"{args.run_name}_success.txt",
        f"{args.run_name}_failed.txt",
        f"{args.run_name}_running.txt",
    ]
    for name in base_files:
        download_file(sftp, f"{remote_logs}/{name}", local_logs / name)

    latest_dir_rel = ""
    latest_path_rel = ""
    latest_dir_ptr = local_logs / f"{args.run_name}_latest_log_dir.txt"
    latest_log_ptr = local_logs / f"{args.run_name}_latest_log_path.txt"
    if latest_dir_ptr.exists():
        latest_dir_rel = latest_dir_ptr.read_text(encoding="utf-8").strip()
    if latest_log_ptr.exists():
        latest_path_rel = latest_log_ptr.read_text(encoding="utf-8").strip()

    if latest_dir_rel:
        remote_dir = posixpath.join(args.remote_repo.rstrip("/"), latest_dir_rel)
        local_dir = repo_root / latest_dir_rel
        download_dir(sftp, remote_dir, local_dir)

    if latest_path_rel:
        remote_file = posixpath.join(args.remote_repo.rstrip("/"), latest_path_rel)
        local_file = repo_root / latest_path_rel
        download_file(sftp, remote_file, local_file)

    sftp.close()
    client.close()

    git_status = subprocess.run(
        ["git", "status", "--short", "outputs/logs"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if not git_status.stdout.strip():
        print("no local log changes to commit")
        return 0

    subprocess.run(["git", "add", "outputs/logs"], cwd=repo_root, check=True)
    commit = subprocess.run(
        ["git", "commit", "-m", f"chore: sync {args.run_name} status"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
        print(commit.stdout)
        print(commit.stderr, file=sys.stderr)
        return commit.returncode
    if args.push:
        subprocess.run(["git", "push", "origin", "master"], cwd=repo_root, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--remote-repo", default="/root/nemotron")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()

    while True:
        code = sync_once(args)
        if code != 0 or not args.loop:
            return code
        print(f"sleeping {args.interval_seconds}s before next sync", flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
