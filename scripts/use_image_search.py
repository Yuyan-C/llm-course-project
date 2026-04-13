import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.tools.search.web_search import run_web_image_search
from src.inquire.web_image_quality import (
    WebImageQualityConfig,
    filter_record_images,
)

from datasets import load_dataset
import numpy as np


def download_image(url: str, output_dir: Path, filename: str) -> Path | None:
    output_path = output_dir / filename
    try:
        request = Request(url, headers={"User-Agent": "llm-course-project/1.0"})
        with urlopen(request, timeout=20) as response:
            content = response.read()
        output_path.write_bytes(content)
        return output_path
    except Exception:
        return None


def infer_extension(url: str) -> str:
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if 1 <= len(ext) <= 5:
            return ext
    return "jpg"


def fetch_images_for_query(
    query: str,
    output_dir: Path,
    max_results: int,
) -> dict:
    tool_result = run_web_image_search(
        query,
        max_results=max_results,
        region="us",
        safesearch="on",
        backend="v2",
    )
    urls = [item.get("image") for item in tool_result.get("results", []) if item.get("image")]

    image_paths = []
    for idx, url in enumerate(urls, start=1):
        ext = infer_extension(url)
        safe_query = re.sub(r"[^a-z0-9_-]+", "_", query.lower()).strip("_")
        filename = f"{safe_query}_{idx}.{ext}"
        saved = download_image(url, output_dir, filename)
        if saved:
            image_paths.append(str(saved))

    return {
        "query": query,
        "image_urls": urls,
        "image_paths": image_paths,
        "tool_result": tool_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and quality-filter web images for INQUIRE queries.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default="output/web_images")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--keep-k", type=int, default=8)
    parser.add_argument("--min-keep", type=int, default=2)
    parser.add_argument("--disable-quality-filter", action="store_true")
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.8)
    parser.add_argument("--stock-thumbnail-short-side", type=int, default=600)
    parser.add_argument("--metadata-relevance-threshold", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if args.split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    quality_config = WebImageQualityConfig(
        min_short_side=args.min_short_side,
        max_aspect_ratio=args.max_aspect_ratio,
        stock_thumbnail_short_side=args.stock_thumbnail_short_side,
        metadata_relevance_threshold=args.metadata_relevance_threshold,
        drop_stock_thumbnails=True,
    )

    all_metadata = []
    for i, query_text in enumerate(queries, start=1):
        payload = fetch_images_for_query(
            query_text,
            output_dir=output_dir,
            max_results=args.max_results,
        )

        if args.disable_quality_filter:
            keep_target = max(args.keep_k, args.min_keep)
            payload["image_paths"] = payload["image_paths"][: keep_target]
        else:
            kept_paths, accepted, rejected = filter_record_images(
                payload,
                quality_config,
                max_keep=args.keep_k,
                min_keep=args.min_keep,
            )
            payload["image_paths_raw"] = payload["image_paths"]
            payload["image_paths"] = kept_paths
            payload["image_quality_accepted"] = accepted
            payload["image_quality_rejected"] = rejected

        all_metadata.append(payload)
        print(
            f"[{i}/{len(queries)}] kept={len(payload['image_paths'])} "
            f"downloaded={len(payload.get('image_paths_raw', payload['image_paths']))} "
            f"query={query_text}"
        )

    metadata_path = output_dir / f"image_search_metadata_{args.split}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=True)

    print(f"Saved metadata to {metadata_path}")
