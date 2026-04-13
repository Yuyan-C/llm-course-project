"""Evaluate image-to-image reranking using downloaded web exemplars."""

import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from PIL import Image

from src.inquire.utils import load_clip
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.web_image_quality import (
    WebImageQualityConfig,
    extract_image_embedding_tensor,
    filter_record_images,
)


def patch_transformers_tokenizer_compat() -> None:
    # all_clip expects transformers tokenizers to provide batch_encode_plus.
    try:
        from transformers import T5Tokenizer, T5TokenizerFast  # type: ignore
    except Exception:
        return

    for cls in (T5Tokenizer, T5TokenizerFast):
        if cls is None:
            continue
        if hasattr(cls, "batch_encode_plus"):
            continue

        def _batch_encode_plus(self, *args, **kwargs):
            return self(*args, **kwargs)

        setattr(cls, "batch_encode_plus", _batch_encode_plus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation with web-image exemplars.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/network/scratch/y/yuyan.chen/inquire/web_images/image_search_metadata_test.json",
        help="Path to downloaded image metadata JSON.",
    )
    parser.add_argument("--save-results-path", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to evaluate (e.g. siglip-so400m-14-384).",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-web-images-per-query", type=int, default=8)
    parser.add_argument("--min-web-images-per-query", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-load-retries", type=int, default=3)
    parser.add_argument("--model-load-retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "topk_mean"],
        help="How to aggregate similarity across web exemplars for each candidate image.",
    )
    parser.add_argument(
        "--top-k-sims",
        type=int,
        default=3,
        help="When using top-k aggregation, keep the top-k exemplar similarities per candidate.",
    )
    parser.add_argument(
        "--apply-web-image-filters",
        action="store_true",
        help="Apply quality filters to downloaded web images before reranking.",
    )
    parser.add_argument(
        "--filtered-metadata-out",
        type=str,
        default=None,
        help="Optional output path for metadata after filtering decisions.",
    )
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.8)
    parser.add_argument("--stock-thumbnail-short-side", type=int, default=600)
    parser.add_argument("--metadata-relevance-threshold", type=float, default=0.25)
    return parser.parse_args()


def normalize_metadata(records_obj: object) -> list[dict]:
    if isinstance(records_obj, list):
        return [item for item in records_obj if isinstance(item, dict)]
    if isinstance(records_obj, dict):
        return [records_obj]
    raise ValueError(f"Expected metadata JSON to be a list or dict, got {type(records_obj).__name__}")


class TimmImageEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        out = self.model(image)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def load_image_encoder(model_name: str, device: str) -> tuple[torch.nn.Module, object]:
    if model_name.startswith("timm:"):
        import timm
        from timm.data import create_transform, resolve_model_data_config

        timm_name = model_name.split(":", 1)[1]
        model = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        model = model.to(device).eval()
        data_config = resolve_model_data_config(model)
        preprocess = create_transform(**data_config, is_training=False)
        return TimmImageEncoder(model), preprocess

    model, preprocess, _ = load_clip(model_name, use_jit=False, device=device)
    return model, preprocess


def aggregate_similarity_scores(
    sim_matrix: torch.Tensor,
    *,
    mode: str,
    top_k: int,
) -> torch.Tensor:
    if sim_matrix.numel() == 0:
        return torch.empty((sim_matrix.shape[0],), dtype=sim_matrix.dtype, device=sim_matrix.device)

    if mode == "mean":
        return sim_matrix.mean(dim=1)

    k = sim_matrix.shape[1] if top_k <= 0 else min(top_k, sim_matrix.shape[1])
    source = sim_matrix
    topk_vals = source.topk(k, dim=1).values
    return topk_vals.mean(dim=1)


def build_query_to_downloads(
    metadata_records: list[dict],
    *,
    quality_config: WebImageQualityConfig,
    max_web_images_per_query: int | None,
    min_web_images_per_query: int | None,
    apply_filters: bool,
) -> tuple[dict[str, list[str]], list[dict], int, int]:
    query_to_downloads: dict[str, list[str]] = {}
    filtered_records: list[dict] = []
    total_candidates = 0
    total_kept = 0

    for record in metadata_records:
        query_text = record.get("query")
        if not isinstance(query_text, str) or not query_text:
            continue

        original_paths = [p for p in record.get("image_paths", []) if isinstance(p, str)]
        total_candidates += len(original_paths)
        out_record = dict(record)

        if apply_filters:
            kept_paths, accepted, rejected = filter_record_images(
                record,
                quality_config,
                max_keep=max_web_images_per_query,
                min_keep=min_web_images_per_query,
            )
            out_record["image_paths_raw"] = original_paths
            out_record["image_paths"] = kept_paths
            out_record["image_quality_accepted"] = accepted
            out_record["image_quality_rejected"] = rejected
            final_paths = kept_paths
        else:
            if max_web_images_per_query is not None and max_web_images_per_query >= 0:
                final_paths = original_paths[:max_web_images_per_query]
            else:
                final_paths = original_paths
            out_record["image_paths"] = final_paths

        total_kept += len(final_paths)
        query_to_downloads.setdefault(query_text, []).extend(final_paths)
        filtered_records.append(out_record)

    for query_text, entries in list(query_to_downloads.items()):
        dedup_entries: list[str] = []
        seen: set[str] = set()
        for path in entries:
            if path in seen:
                continue
            seen.add(path)
            dedup_entries.append(path)
        query_to_downloads[query_text] = dedup_entries

    return query_to_downloads, filtered_records, total_candidates, total_kept


def main() -> None:
    args = parse_args()
    patch_transformers_tokenizer_compat()

    split = args.split
    save_results_path = args.save_results_path or f"results_rerank_with_clip_{split}.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    with Path(args.metadata).open("r", encoding="utf-8") as f:
        metadata_raw = json.load(f)
    metadata_records = normalize_metadata(metadata_raw)

    quality_config = WebImageQualityConfig(
        min_short_side=args.min_short_side,
        max_aspect_ratio=args.max_aspect_ratio,
        stock_thumbnail_short_side=args.stock_thumbnail_short_side,
        metadata_relevance_threshold=args.metadata_relevance_threshold,
        drop_stock_thumbnails=True,
    )

    query_to_downloads, filtered_records, total_candidates, total_kept = build_query_to_downloads(
        metadata_records,
        quality_config=quality_config,
        max_web_images_per_query=args.max_web_images_per_query,
        min_web_images_per_query=args.min_web_images_per_query,
        apply_filters=args.apply_web_image_filters,
    )
    print(
        f"Web image candidates: {total_candidates}, kept: {total_kept}, "
        f"queries with web images: {sum(1 for q in query_to_downloads if query_to_downloads[q])}"
    )

    if args.filtered_metadata_out:
        out_path = Path(args.filtered_metadata_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(filtered_records, f, indent=2, ensure_ascii=True)
        print(f"Saved filtered metadata to {out_path}")

    all_models_available = {
        "vit-b-32": "hf_clip:openai/clip-vit-base-patch32",
        "bioclip": "bioclip",
        "biocap": "biocap",
        "dinov3-b16": "timm:vit_base_patch16_dinov3",
        "siglip-so400m-14-384": "open_clip:ViT-SO400M-14-SigLIP-384/webli",
    }

    if args.models:
        unknown = [name for name in args.models if name not in all_models_available]
        if unknown:
            raise ValueError(f"Unknown model key(s): {unknown}. Valid keys: {list(all_models_available.keys())}")
        all_models = {name: all_models_available[name] for name in args.models}
    else:
        all_models = all_models_available

    batch_size = args.batch_size
    num_workers = args.num_workers
    results = []

    for title, clip_name in all_models.items():
        model = preprocess = None
        last_exc: Exception | None = None
        for attempt in range(1, args.model_load_retries + 2):
            try:
                model, preprocess = load_image_encoder(clip_name, device=device)
                break
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
                if attempt > args.model_load_retries:
                    break
                print(
                    f"[{title}] load failed on attempt {attempt}/{args.model_load_retries + 1}: "
                    f"{exc.__class__.__name__}: {exc}. Retrying in {args.model_load_retry_sleep:.1f}s..."
                )
                time.sleep(args.model_load_retry_sleep)
        if model is None or preprocess is None:
            if last_exc is None:
                raise RuntimeError(f"Failed to load model for {title} with unknown error.")
            raise last_exc

        def collate_transform(examples):
            pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
            ids = [ex["inat24_image_id"] for ex in examples]
            return pixel_values, ids

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_transform,
            num_workers=num_workers,
        )

        image_emb_cache = {}
        for images, ids in tqdm(dataloader, total=len(dataset) // batch_size):
            with torch.no_grad(), torch.autocast(device):
                image_embs = extract_image_embedding_tensor(model.encode_image(images.to(device))).float().cpu()
                image_embs /= image_embs.norm(dim=-1, keepdim=True)
            image_emb_cache.update(dict(zip(ids, image_embs)))

        def embed_downloaded_images(paths: list[str]) -> torch.Tensor:
            images = []
            for path in paths:
                try:
                    with Image.open(path) as img:
                        images.append(preprocess(img.convert("RGB")).unsqueeze(0))
                except Exception:
                    continue
            if not images:
                dim = image_emb_cache[next(iter(image_emb_cache))].shape[-1]
                return torch.empty((0, dim))
            batch = torch.cat(images)
            with torch.no_grad(), torch.autocast(device):
                emb = extract_image_embedding_tensor(model.encode_image(batch.to(device))).float().cpu()
                emb /= emb.norm(dim=-1, keepdim=True)
            return emb

        metrics_avg = MetricAverage()
        evaluated_queries = 0
        for query in queries:
            query_ds = dataset.select(np.argwhere(np.asarray(dataset["query"]) == query).squeeze())

            download_paths = [p for p in query_to_downloads.get(query, []) if Path(p).exists()]
            download_embs = embed_downloaded_images(download_paths)
            if download_embs.numel() == 0:
                continue

            image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds["inat24_image_id"]])
            sim_matrix = image_embs.float() @ download_embs.float().T
            y_pred = aggregate_similarity_scores(
                sim_matrix,
                mode=args.aggregation,
                top_k=args.top_k_sims,
            ).numpy()
            y_true = np.asarray(query_ds["relevant"])

            pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
            metrics_avg.update([ap * 100, ndcg * 100, mrr])
            results.append(dict(model=title, query=query, ap=ap * 100, ndcg=ndcg * 100, mrr=mrr))
            evaluated_queries += 1

        if metrics_avg.avg is None:
            print(f"{title:30s}\tno valid queries after filtering")
        else:
            ap, ndcg, mrr = metrics_avg.avg
            print(f"{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}\tqueries={evaluated_queries}")

    results_df = pd.DataFrame.from_dict(results)
    pd.options.display.float_format = " {:,.2f}".format
    if len(results_df) == 0:
        print("No evaluation rows were produced. Check metadata/filter settings.")
    else:
        print(results_df.groupby("model").agg({"ap": "mean", "ndcg": "mean", "mrr": "mean"}).sort_values("ap"))

    results_df.to_csv(save_results_path)
    print("All done! Saved results to", save_results_path)


if __name__ == "__main__":
    main()
