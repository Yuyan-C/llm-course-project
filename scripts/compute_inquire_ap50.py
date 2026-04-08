#!/usr/bin/env python3
"""Compute mean AP@50 by INQUIRE prompt type (supercategory) per model."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CATEGORY_ORDER = ["Appearance", "Behavior", "Context", "Species"]


def load_query_map(queries_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with queries_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("query_text") or "").strip()
            cat = (row.get("supercategory") or "").strip()
            if text:
                mapping[text] = cat
    return mapping


def read_results(results_csv: Path) -> Iterable[Tuple[str, str, float]]:
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("model") or "").strip()
            query = (row.get("query") or row.get("query_text") or "").strip()
            ap_raw = row.get("ap")
            if not model or not query or ap_raw is None:
                continue
            try:
                ap_val = float(ap_raw)
            except ValueError:
                continue
            yield model, query, ap_val


def compute_means(results_csv: Path, queries_csv: Path):
    query_to_cat = load_query_map(queries_csv)
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    missing = 0
    for model, query, ap in read_results(results_csv):
        cat = query_to_cat.get(query)
        if not cat:
            missing += 1
            continue
        grouped[(model, cat)].append(ap)
    return grouped, missing


def write_long_csv(out_path: Path, grouped: Dict[Tuple[str, str], List[float]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "prompt_type", "ap50_mean", "n"])
        for (model, cat), values in sorted(grouped.items()):
            mean = sum(values) / max(len(values), 1)
            writer.writerow([model, cat, f"{mean:.4f}", len(values)])


def print_table(grouped: Dict[Tuple[str, str], List[float]]):
    models = sorted({m for (m, _) in grouped.keys()})
    header = ["Model"] + CATEGORY_ORDER
    widths = [max(len(h), 8) for h in header]
    rows = []
    for model in models:
        row = [model]
        for cat in CATEGORY_ORDER:
            values = grouped.get((model, cat))
            if values:
                row.append(f"{sum(values)/len(values):.1f}")
            else:
                row.append("-")
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mean AP@50 by INQUIRE prompt type per model.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/llm-course-project/results_rerank_with_clip_test.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("/network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/llm-course-project/results_ap50_by_prompt.csv"),
    )
    args = parser.parse_args()

    grouped, missing = compute_means(args.results, args.queries)
    write_long_csv(args.out, grouped)
    print_table(grouped)
    if missing:
        print(f"\nWarning: {missing} result rows did not match a query in {args.queries}")


if __name__ == "__main__":
    main()
