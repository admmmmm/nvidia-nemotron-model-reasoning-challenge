from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "nemotron-site-scraper/1.0"
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public JSONL data from nemotron.huikang.dev.")
    parser.add_argument("--base-url", default="https://nemotron.huikang.dev")
    parser.add_argument("--output-dir", default="data/external/nemotron_huikang_dev")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw token trace downloads.")
    parser.add_argument("--limit-problems", type=int, default=0, help="For testing only.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def save_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8", newline="")


def download_if_needed(
    session: requests.Session,
    base_url: str,
    relative_url: str,
    target_path: Path,
    overwrite: bool,
) -> tuple[str, int, bool]:
    if target_path.exists() and not overwrite:
        return relative_url, target_path.stat().st_size, False
    text = fetch_text(session, f"{base_url}/{relative_url}")
    save_text(target_path, text)
    return relative_url, target_path.stat().st_size, True


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_generation(generation_rows: list[dict]) -> dict[str, int]:
    total_runs = 0
    partial = 0
    solved = 0
    unsolved = 0
    for row in generation_rows:
        total_runs += len(row.get("runs", []))
        if row.get("num_runs", 0) == 0:
            unsolved += 1
        elif row.get("any_correct"):
            if all(run.get("correct") for run in row.get("runs", [])):
                solved += 1
            else:
                partial += 1
        else:
            unsolved += 1
    return {
        "problems": len(generation_rows),
        "raw_runs": total_runs,
        "solved": solved,
        "partial": partial,
        "unsolved": unsolved,
    }


def run_download_jobs(
    session: requests.Session,
    base_url: str,
    output_dir: Path,
    jobs: Iterable[tuple[str, Path]],
    workers: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    total_bytes = 0
    lock = threading.Lock()

    def worker(relative_url: str, path: Path) -> tuple[str, int, bool]:
        return download_if_needed(session, base_url, relative_url, path, overwrite)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, relative_url, path) for relative_url, path in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            relative_url, size_bytes, did_download = future.result()
            with lock:
                total_bytes += size_bytes if did_download else 0
                if did_download:
                    downloaded += 1
                else:
                    skipped += 1
            if index % 200 == 0:
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "downloaded": downloaded,
                            "skipped": skipped,
                            "last": relative_url,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    return downloaded, skipped, total_bytes


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    session = build_session()

    top_level_files = [
        "problems.jsonl",
        "generation.jsonl",
        "vocab.jsonl",
    ]

    for filename in top_level_files:
        relative_url = filename
        target_path = output_dir / filename
        _, size_bytes, did_download = download_if_needed(
            session=session,
            base_url=base_url,
            relative_url=relative_url,
            target_path=target_path,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "file": filename,
                    "downloaded": did_download,
                    "bytes": size_bytes,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    problems_rows = load_jsonl(output_dir / "problems.jsonl")
    generation_rows = load_jsonl(output_dir / "generation.jsonl")
    if args.limit_problems > 0:
        selected_ids = {row["id"] for row in problems_rows[: args.limit_problems]}
        problems_rows = [row for row in problems_rows if row["id"] in selected_ids]
        generation_rows = [row for row in generation_rows if row["id"] in selected_ids]

    problem_jobs = []
    for row in problems_rows:
        problem_id = row["id"]
        relative_url = f"problems/{problem_id}.jsonl"
        target_path = output_dir / "problems" / f"{problem_id}.jsonl"
        problem_jobs.append((relative_url, target_path))

    print(
        json.dumps(
            {
                "stage": "problem_details",
                "jobs": len(problem_jobs),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    problem_downloaded, problem_skipped, problem_bytes = run_download_jobs(
        session=session,
        base_url=base_url,
        output_dir=output_dir,
        jobs=problem_jobs,
        workers=args.workers,
        overwrite=args.overwrite,
    )

    raw_jobs = []
    if not args.skip_raw:
        for row in generation_rows:
            problem_id = row["id"]
            for run in row.get("runs", []):
                run_name = run["run"]
                relative_url = f"raw/{problem_id}/{run_name}"
                target_path = output_dir / "raw" / problem_id / run_name
                raw_jobs.append((relative_url, target_path))
        print(
            json.dumps(
                {
                    "stage": "raw_tokens",
                    "jobs": len(raw_jobs),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raw_downloaded, raw_skipped, raw_bytes = run_download_jobs(
            session=session,
            base_url=base_url,
            output_dir=output_dir,
            jobs=raw_jobs,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    else:
        raw_downloaded = raw_skipped = raw_bytes = 0

    summary = {
        "base_url": base_url,
        "output_dir": str(output_dir),
        "problems_index_rows": len(problems_rows),
        "generation_index_rows": len(generation_rows),
        "generation_summary": summarize_generation(generation_rows),
        "problem_details": {
            "downloaded": problem_downloaded,
            "skipped": problem_skipped,
            "bytes_downloaded": problem_bytes,
        },
        "raw_tokens": {
            "downloaded": raw_downloaded,
            "skipped": raw_skipped,
            "bytes_downloaded": raw_bytes,
        },
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    save_text(output_dir / "download_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
