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
    parser.add_argument("--run-folder-name", default="")
    parser.add_argument("--source-repo", default="/root/nemotron")
    parser.add_argument("--git-repo", default="/root/nemotron_git")
    parser.add_argument("--sync-interval-seconds", type=int, default=60)
    args = parser.parse_args()

    run_folder_name = args.run_folder_name or args.run_name

    remote_script = textwrap.dedent(
        f"""
        set -euo pipefail
        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

        echo "=== verify paths ==="
        ls -la /root || true
        echo "--- /root/nemotron/scripts ---"
        ls -la /root/nemotron/scripts || true
        echo "--- /root/nemotron_git/scripts ---"
        ls -la /root/nemotron_git/scripts || true

        if [[ ! -f /root/nemotron/scripts/start_train_background.sh ]] && [[ -f /root/nemotron_git/scripts/start_train_background.sh ]]; then
          cp -f /root/nemotron_git/scripts/start_train_background.sh /root/nemotron/scripts/
          cp -f /root/nemotron_git/scripts/sync_loop.sh /root/nemotron/scripts/
          cp -f /root/nemotron_git/scripts/sync_run_to_git.sh /root/nemotron/scripts/
          chmod +x /root/nemotron/scripts/start_train_background.sh /root/nemotron/scripts/sync_loop.sh /root/nemotron/scripts/sync_run_to_git.sh
        fi

        cd /root/nemotron
        echo "=== final scripts dir ==="
        ls -la scripts || true

        export RUN_NAME="{args.run_name}"
        export RUN_FOLDER_NAME="{run_folder_name}"
        export GIT_SYNC_SOURCE_REPO="{args.source_repo}"
        export GIT_SYNC_REPO="{args.git_repo}"
        export SYNC_INTERVAL_SECONDS="{args.sync_interval_seconds}"

        bash scripts/start_train_background.sh "{run_folder_name}"

        echo "=== process check ==="
        ps -ef | grep -E "run_nemotron_lora|src.train.sft_local|sync_loop.sh" | grep -v grep || true
        echo "=== latest pointers ==="
        cat outputs/logs/${{RUN_NAME}}_latest_log_dir.txt 2>/dev/null || true
        cat outputs/logs/${{RUN_NAME}}_latest_log_path.txt 2>/dev/null || true
        echo "=== launch tail ==="
        tail -n 80 outputs/logs/launch_latest.out 2>/dev/null || true
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

    remote_path = "/root/nemotron/.tmp_remote_start_train.sh"
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
