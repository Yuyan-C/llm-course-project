#!/usr/bin/env python3
"""Summarize baseline-vs-top8 rerank results overall and by supercategory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["ap", "ndcg", "mrr"]
SUPERCATEGORY_ORDER = ["Appearance", "Behavior", "Context", "Species"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline-vs-top8 comparison tables for all models.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        required=True,
        help="CSV path from baseline top3 run.",
    )
    parser.add_argument(
        "--top8-results",
        type=Path,
        required=True,
        help="CSV path from top8 filtered top-k run.",
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
        default=Path("output/web_images/compare/results_all_models_baseline_vs_top8_summary.csv"),
    )
    parser.add_argument(
        "--supercategory-out",
        type=Path,
        default=Path("output/web_images/compare/results_all_models_baseline_vs_top8_by_supercategory.csv"),
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


def summarize_overall(baseline: pd.DataFrame, top8: pd.DataFrame) -> pd.DataFrame:
    baseline_means = baseline.groupby("model", as_index=False)[METRICS].mean().rename(
        columns={m: f"{m}_baseline_top3" for m in METRICS}
    )
    top8_means = top8.groupby("model", as_index=False)[METRICS].mean().rename(
        columns={m: f"{m}_top8_topk" for m in METRICS}
    )

    merged = baseline_means.merge(top8_means, on="model", how="outer")
    merged["delta_ap_topk_vs_base"] = merged["ap_top8_topk"] - merged["ap_baseline_top3"]
    merged["delta_ndcg_topk_vs_base"] = merged["ndcg_top8_topk"] - merged["ndcg_baseline_top3"]
    merged["delta_mrr_topk_vs_base"] = merged["mrr_top8_topk"] - merged["mrr_baseline_top3"]
    merged = merged.sort_values(by="ap_baseline_top3", ascending=False, na_position="last")
    return merged


def summarize_by_supercategory(
    baseline: pd.DataFrame,
    top8: pd.DataFrame,
    query_to_supercategory: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    baseline_joined = baseline.merge(query_to_supercategory, on="query", how="left")
    top8_joined = top8.merge(query_to_supercategory, on="query", how="left")

    missing_baseline = int(baseline_joined["supercategory"].isna().sum())
    missing_top8 = int(top8_joined["supercategory"].isna().sum())

    baseline_joined = baseline_joined.dropna(subset=["supercategory"])
    top8_joined = top8_joined.dropna(subset=["supercategory"])

    baseline_agg = (
        baseline_joined.groupby(["model", "supercategory"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_baseline_top3" for m in METRICS})
    )
    baseline_counts = (
        baseline_joined.groupby(["model", "supercategory"], as_index=False)
        .size()
        .rename(columns={"size": "n_baseline_top3"})
    )
    baseline_agg = baseline_agg.merge(baseline_counts, on=["model", "supercategory"], how="left")

    top8_agg = (
        top8_joined.groupby(["model", "supercategory"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_top8_topk" for m in METRICS})
    )
    top8_counts = (
        top8_joined.groupby(["model", "supercategory"], as_index=False)
        .size()
        .rename(columns={"size": "n_top8_topk"})
    )
    top8_agg = top8_agg.merge(top8_counts, on=["model", "supercategory"], how="left")

    merged = baseline_agg.merge(top8_agg, on=["model", "supercategory"], how="outer")
    merged["delta_ap_topk_vs_base"] = merged["ap_top8_topk"] - merged["ap_baseline_top3"]
    merged["delta_ndcg_topk_vs_base"] = merged["ndcg_top8_topk"] - merged["ndcg_baseline_top3"]
    merged["delta_mrr_topk_vs_base"] = merged["mrr_top8_topk"] - merged["mrr_baseline_top3"]

    merged["supercategory"] = pd.Categorical(
        merged["supercategory"],
        categories=SUPERCATEGORY_ORDER,
        ordered=True,
    )
    merged = merged.sort_values(by=["model", "supercategory"], na_position="last")
    merged["supercategory"] = merged["supercategory"].astype(str)
    return merged, missing_baseline, missing_top8


def main() -> None:
    args = parse_args()
    baseline = load_results(args.baseline_results)
    top8 = load_results(args.top8_results)
    query_to_supercategory = load_query_supercategories(args.queries_csv)

    overall = summarize_overall(baseline, top8)
    by_supercategory, missing_baseline, missing_top8 = summarize_by_supercategory(
        baseline,
        top8,
        query_to_supercategory,
    )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.supercategory_out.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.summary_out, index=False)
    by_supercategory.to_csv(args.supercategory_out, index=False)

    print(f"Wrote overall summary: {args.summary_out}")
    print(f"Wrote supercategory summary: {args.supercategory_out}")
    if missing_baseline or missing_top8:
        print(
            "Warning: missing supercategory mapping rows "
            f"(baseline={missing_baseline}, top8={missing_top8})"
        )


if __name__ == "__main__":
    main()
