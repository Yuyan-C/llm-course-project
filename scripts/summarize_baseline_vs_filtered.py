#!/usr/bin/env python3
"""Summarize baseline-vs-filtered rerank results overall and by supercategory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["ap", "ndcg", "mrr"]
SUPERCATEGORY_ORDER = ["Appearance", "Behavior", "Context", "Species"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline-vs-filtered comparison tables for all models.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        required=True,
        help="CSV path from baseline run.",
    )
    parser.add_argument(
        "--comparison-results",
        type=Path,
        required=True,
        help="CSV path from filtered run.",
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        required=True,
        help="CSV with query_text and supercategory columns.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("output/web_images/compare/results_all_models_baseline_vs_filtered_summary.csv"),
    )
    parser.add_argument(
        "--supercategory-out",
        type=Path,
        default=Path("output/web_images/compare/results_all_models_baseline_vs_filtered_by_supercategory.csv"),
    )
    return parser.parse_args()


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


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


def summarize_overall(baseline: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    baseline_means = baseline.groupby("model", as_index=False)[METRICS].mean().rename(
        columns={m: f"{m}_baseline" for m in METRICS}
    )
    filtered_means = filtered.groupby("model", as_index=False)[METRICS].mean().rename(
        columns={m: f"{m}_filtered" for m in METRICS}
    )

    merged = baseline_means.merge(filtered_means, on="model", how="outer")
    merged["delta_ap_filtered_vs_baseline"] = merged["ap_filtered"] - merged["ap_baseline"]
    merged["delta_ndcg_filtered_vs_baseline"] = merged["ndcg_filtered"] - merged["ndcg_baseline"]
    merged["delta_mrr_filtered_vs_baseline"] = merged["mrr_filtered"] - merged["mrr_baseline"]
    merged = merged.sort_values(by="ap_baseline", ascending=False, na_position="last")
    return merged


def summarize_by_supercategory(
    baseline: pd.DataFrame,
    filtered: pd.DataFrame,
    query_to_supercategory: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    baseline_joined = baseline.merge(query_to_supercategory, on="query", how="left")
    filtered_joined = filtered.merge(query_to_supercategory, on="query", how="left")

    missing_baseline = int(baseline_joined["supercategory"].isna().sum())
    missing_filtered = int(filtered_joined["supercategory"].isna().sum())

    baseline_joined = baseline_joined.dropna(subset=["supercategory"])
    filtered_joined = filtered_joined.dropna(subset=["supercategory"])

    baseline_agg = (
        baseline_joined.groupby(["model", "supercategory"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_baseline" for m in METRICS})
    )
    baseline_counts = (
        baseline_joined.groupby(["model", "supercategory"], as_index=False)
        .size()
        .rename(columns={"size": "n_baseline"})
    )
    baseline_agg = baseline_agg.merge(baseline_counts, on=["model", "supercategory"], how="left")

    filtered_agg = (
        filtered_joined.groupby(["model", "supercategory"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_filtered" for m in METRICS})
    )
    filtered_counts = (
        filtered_joined.groupby(["model", "supercategory"], as_index=False)
        .size()
        .rename(columns={"size": "n_filtered"})
    )
    filtered_agg = filtered_agg.merge(filtered_counts, on=["model", "supercategory"], how="left")

    merged = baseline_agg.merge(filtered_agg, on=["model", "supercategory"], how="outer")
    merged["delta_ap_filtered_vs_baseline"] = merged["ap_filtered"] - merged["ap_baseline"]
    merged["delta_ndcg_filtered_vs_baseline"] = merged["ndcg_filtered"] - merged["ndcg_baseline"]
    merged["delta_mrr_filtered_vs_baseline"] = merged["mrr_filtered"] - merged["mrr_baseline"]

    merged["supercategory"] = pd.Categorical(
        merged["supercategory"],
        categories=SUPERCATEGORY_ORDER,
        ordered=True,
    )
    merged = merged.sort_values(by=["model", "supercategory"], na_position="last")
    merged["supercategory"] = merged["supercategory"].astype(str)
    return merged, missing_baseline, missing_filtered


def main() -> None:
    args = parse_args()
    baseline = load_results(args.baseline_results)
    filtered = load_results(args.comparison_results)
    query_to_supercategory = load_query_supercategories(args.queries_csv)

    overall = summarize_overall(baseline, filtered)
    by_supercategory, missing_baseline, missing_filtered = summarize_by_supercategory(
        baseline,
        filtered,
        query_to_supercategory,
    )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.supercategory_out.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.summary_out, index=False)
    by_supercategory.to_csv(args.supercategory_out, index=False)

    print(f"Wrote overall summary: {args.summary_out}")
    print(f"Wrote supercategory summary: {args.supercategory_out}")
    if missing_baseline or missing_filtered:
        print(
            "Warning: missing supercategory mapping rows "
            f"(baseline={missing_baseline}, filtered={missing_filtered})"
        )


if __name__ == "__main__":
    main()
