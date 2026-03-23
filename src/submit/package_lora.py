from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a LoRA adapter into submission.zip.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-file", default="outputs/submissions/submission.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")

    present_files = {path.name for path in adapter_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - present_files)
    if missing:
        raise SystemExit(
            "Adapter directory is missing required files: "
            + ", ".join(missing)
        )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_file, "w", compression=ZIP_DEFLATED) as archive:
        for path in adapter_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(adapter_dir))

    print(f"Created {output_file}")


if __name__ == "__main__":
    main()
