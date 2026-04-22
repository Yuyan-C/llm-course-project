"""Merge caption-cache shard JSON files into a single cache JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge caption cache shards.")
    parser.add_argument(
        "--input-template",
        type=str,
        required=True,
        help="Input path template containing '{i}' placeholder for shard index.",
    )
    parser.add_argument(
        "--fallback-input-template",
        type=str,
        default=None,
        help="Optional fallback template used when primary shard file is missing.",
    )
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")

    merged: dict[str, str] = {}
    for i in range(args.num_shards):
        path = Path(args.input_template.format(i=i))
        if not path.exists() and args.fallback_input_template:
            fallback = Path(args.fallback_input_template.format(i=i))
            if fallback.exists():
                path = fallback
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
        merged.update({str(k): str(v) for k, v in payload.items()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=True)
    print(f"Merged {args.num_shards} shards into {output} ({len(merged)} entries).")


if __name__ == "__main__":
    main()
