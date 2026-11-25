"""Create visible train/test folder views without changing the data pipeline.

The canonical files remain under data/raw and data/processed. This utility
copies processed videos into data/split_views/{train,val,test}/{normal,violence}
using the split recorded in data/metadata/local/manifest.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/local/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/split_views"))
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    manifest = (root / args.manifest).resolve()
    output = (root / args.output).resolve()
    if root not in manifest.parents or root not in output.parents:
        parser.error("manifest and output must stay inside the repository")
    if not manifest.is_file():
        parser.error(f"manifest not found: {manifest}")

    selected = set(args.split or ("train", "val", "test"))
    copied = 0
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        label = row.get("label")
        if split not in selected or label not in {"normal", "violence"}:
            continue
        source_value = row.get("processed_path") or row.get("raw_path")
        if not source_value:
            raise SystemExit(f"manifest row has no video path: {row.get('sample_id')}")
        source = (root / source_value).resolve()
        if root not in source.parents or not source.is_file():
            raise SystemExit(f"video not found for {row.get('sample_id')}: {source}")
        destination = output / split / label / f"{row['sample_id']}{source.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    print(f"Materialized {copied} video view(s) under {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
