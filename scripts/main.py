from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any, List

import yaml
from datasets import load_dataset
import numpy as np

from src.reranking import EcologicalRerankingOrchestrator


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def run_reranking_from_loaded_images(
    prompt: str,
    image_paths: List[str],
    raw_images: List[Any],
    top_k: int = 50,
    config_path: str = "configs/eval.yml",
):
    with open(config_path, "r") as f:
        config = to_namespace(yaml.safe_load(f))

    clip_model_name = getattr(config, "clip_model_name", "bioclip")
    detector_threshold = getattr(config, "detector_threshold", 0.30)
    bioclip_threshold = getattr(config, "bioclip_threshold", 0.10)

    orchestrator = EcologicalRerankingOrchestrator(
        clip_model_name=clip_model_name,
        detector_threshold=detector_threshold,
        bioclip_threshold=bioclip_threshold,
    )
    return orchestrator.run(prompt=prompt, image_paths=image_paths, raw_images=raw_images, top_k=top_k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo runner for loaded-image reranking.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--query_index", type=int, default=0, help="Query index from INQUIRE-Rerank split.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_json", type=str, default="rerank_results.json")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if args.split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()
    query_text = queries[args.query_index]
    query_ds = dataset.select(np.argwhere(np.asarray(dataset["query"]) == query_text).squeeze())

    image_paths = query_ds["inat24_file_name"]
    raw_images = query_ds["image"]

    result = run_reranking_from_loaded_images(
        prompt=query_text,
        image_paths=image_paths,
        raw_images=raw_images,
        top_k=args.top_k,
    )

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "plan": result.plan,
                "stats": result.stats,
                "ranked_images": result.ranked_images,
            },
            f,
            indent=2,
        )

    print("=" * 40)
    print("Plan:")
    print(result.plan)
    print("=" * 40)
    print("Stats:")
    print(result.stats)
    print("=" * 40)
    print("Top results:")
    for row in result.ranked_images[: min(5, len(result.ranked_images))]:
        print(row)
    print("=" * 40)
    print(f"Saved results to {args.output_json}")