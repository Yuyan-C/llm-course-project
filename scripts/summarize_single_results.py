#!/usr/bin/env python3
"""Summarize a single rerank-results CSV overall and by supercategory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["ap", "ndcg", "mrr"]
SUPERCATEGORY_ORDER = ["Appearance", "Behavior", "Context", "Species"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build overall and supercategory tables for one results CSV.",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--queries-csv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--supercategory-out", type=Path, required=True)
    return parser.parse_args()


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    return df.drop(columns=unnamed) if unnamed else df


def load_results(path: Path) -> pd.DataFrame:
    df = _drop_unnamed(pd.read_csv(path))
    query_col = "query" if "query" in df.columns else ("query_text" if "query_text" in df.columns else None)
    if query_col is None:
        raise ValueError(f"{path} is missing query/query_text column")

    required = {"model", query_col, *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    out = df[["model", query_col, *METRICS]].copy()
    out = out.rename(columns={query_col: "query"})
    out["model"] = out["model"].astype(str)
    out["query"] = out["query"].astype(str)
    return out


def load_query_supercategories(path: Path) -> pd.DataFrame:
    df = _drop_unnamed(pd.read_csv(path))
    required = {"query_text", "supercategory"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    out = df[["query_text", "supercategory"]].copy()
    out = out.rename(columns={"query_text": "query"})
    out["query"] = out["query"].astype(str)
    out["supercategory"] = out["supercategory"].astype(str)
    out = out.drop_duplicates(subset=["query"])
    return out


def summarize_overall(results: pd.DataFrame) -> pd.DataFrame:
    metric_means = results.groupby("model", as_index=False)[METRICS].mean()
    counts = results.groupby("model", as_index=False).size().rename(columns={"size": "n_queries"})
    merged = metric_means.merge(counts, on="model", how="left")
    return merged.sort_values(by="ap", ascending=False, na_position="last")


def summarize_by_supercategory(
    results: pd.DataFrame,
    query_to_supercategory: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    joined = results.merge(query_to_supercategory, on="query", how="left")
    missing = int(joined["supercategory"].isna().sum())
    joined = joined.dropna(subset=["supercategory"])

    metric_means = joined.groupby(["model", "supercategory"], as_index=False)[METRICS].mean()
    counts = joined.groupby(["model", "supercategory"], as_index=False).size().rename(columns={"size": "n_queries"})
    merged = metric_means.merge(counts, on=["model", "supercategory"], how="left")

    merged["supercategory"] = pd.Categorical(
        merged["supercategory"],
        categories=SUPERCATEGORY_ORDER,
        ordered=True,
    )
    merged = merged.sort_values(by=["model", "supercategory"], na_position="last")
    merged["supercategory"] = merged["supercategory"].astype(str)
    return merged, missing


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    query_to_supercategory = load_query_supercategories(args.queries_csv)

    overall = summarize_overall(results)
    by_supercategory, missing = summarize_by_supercategory(results, query_to_supercategory)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.supercategory_out.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.summary_out, index=False)
    by_supercategory.to_csv(args.supercategory_out, index=False)

    print(f"Wrote overall summary: {args.summary_out}")
    print(f"Wrote supercategory summary: {args.supercategory_out}")
    if missing:
        print(f"Warning: missing supercategory mapping rows: {missing}")


if __name__ == "__main__":
    main()
