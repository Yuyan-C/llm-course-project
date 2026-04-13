"""Evaluate reranking in text space using repo VLM captions and web metadata text."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from PIL import Image

from src.inquire.lmm_utils_new import ModelWrapper
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.utils import load_clip
from src.inquire.web_image_quality import WebImageQualityConfig, filter_record_images


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
    parser = argparse.ArgumentParser(description="Run text-to-text reranking with VLM captions.")
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
        help="Subset of retrieval model keys to evaluate (e.g. siglip-so400m-14-384).",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-web-images-per-query", type=int, default=8)
    parser.add_argument("--min-web-images-per-query", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
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
        help="Apply metadata-based quality filters to downloaded web images before reranking.",
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

    parser.add_argument(
        "--caption-vlm-model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Repo VLM model name used to caption test-set images.",
    )
    parser.add_argument(
        "--caption-prompt",
        type=str,
        default=(
            "Describe this image in one concise factual sentence. "
            "Focus on species, appearance, behavior, and context."
        ),
    )
    parser.add_argument("--caption-max-new-tokens", type=int, default=48)
    parser.add_argument("--caption-temperature", type=float, default=0.0)
    parser.add_argument("--caption-load-retries", type=int, default=3)
    parser.add_argument("--caption-load-retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--caption-cache-path",
        type=str,
        default=None,
        help="JSON cache path for generated captions. Defaults to output/web_images/compare.",
    )
    parser.add_argument(
        "--force-regenerate-captions",
        action="store_true",
        help="Ignore existing caption cache and regenerate captions.",
    )
    return parser.parse_args()


def normalize_metadata(records_obj: object) -> list[dict]:
    if isinstance(records_obj, list):
        return [item for item in records_obj if isinstance(item, dict)]
    if isinstance(records_obj, dict):
        return [records_obj]
    raise ValueError(f"Expected metadata JSON to be a list or dict, got {type(records_obj).__name__}")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_url_text(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path = re.sub(r"[^a-zA-Z0-9]+", " ", parsed.path)
    return _normalize_whitespace(f"{host} {path}")


def _search_item_by_url(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    tool_result = record.get("tool_result")
    if not isinstance(tool_result, dict):
        return out
    items = tool_result.get("results", [])
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("image") or item.get("url")
        if not isinstance(url, str) or not url:
            continue
        out[url] = item
    return out


def _fallback_descriptor_from_path(path_str: str) -> str:
    stem = Path(path_str).stem
    return _normalize_whitespace(re.sub(r"[_-]+", " ", stem))


def _descriptor_from_fields(
    *,
    path: str,
    title: str | None,
    source: str | None,
    url: str | None,
    thumbnail: str | None,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    if source:
        parts.append(source)
    if url:
        parts.append(_clean_url_text(url))
    if thumbnail:
        parts.append(_clean_url_text(thumbnail))

    text = _normalize_whitespace(" ".join(parts))
    if text:
        return text
    return _fallback_descriptor_from_path(path)


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
    return sim_matrix.topk(k, dim=1).values.mean(dim=1)


def _to_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        return Image.fromarray(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type for captioning: {type(image_obj).__name__}")


def _extract_embedding_tensor(model_output: Any, primary_attr: str) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output

    for attr in (primary_attr, "pooler_output", "last_hidden_state"):
        value = getattr(model_output, attr, None)
        if torch.is_tensor(value):
            if attr == "last_hidden_state" and value.ndim >= 3:
                return value[:, 0, :]
            return value

    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        first = model_output[0]
        if torch.is_tensor(first):
            if first.ndim >= 3:
                return first[:, 0, :]
            return first

    raise TypeError(
        f"Unsupported embedding output type for '{primary_attr}': {type(model_output).__name__}"
    )


def extract_text_embedding_tensor(model_output: Any) -> torch.Tensor:
    return _extract_embedding_tensor(model_output, "text_embeds")


def load_caption_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Caption cache must be a dict JSON object: {path}")
    return {str(k): str(v) for k, v in payload.items()}


def save_caption_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=True)


def _caption_key(image_id: Any, caption_model: str) -> str:
    return f"{caption_model}::inat24:{image_id}"


def load_caption_model_with_retries(args: argparse.Namespace, *, device: str) -> ModelWrapper:
    last_exc: Exception | None = None
    for attempt in range(1, args.caption_load_retries + 2):
        try:
            return ModelWrapper(args.caption_vlm_model, device=device)
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > args.caption_load_retries:
                break
            print(
                f"[caption-model] load failed on attempt {attempt}/{args.caption_load_retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {args.caption_load_retry_sleep:.1f}s..."
            )
            time.sleep(args.caption_load_retry_sleep)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load caption model with unknown error.")


def generate_missing_captions(
    *,
    image_ids: list[Any],
    caption_cache: dict[str, str],
    force_regenerate: bool,
    caption_model_name: str,
    caption_wrapper: ModelWrapper,
    dataset: Any,
    image_id_to_index: dict[Any, int],
    caption_prompt: str,
    caption_max_new_tokens: int,
    caption_temperature: float,
) -> None:
    missing_ids: list[Any] = []
    for image_id in dict.fromkeys(image_ids):
        key = _caption_key(image_id, caption_model_name)
        if (not force_regenerate) and key in caption_cache and caption_cache[key]:
            continue
        if image_id in image_id_to_index:
            missing_ids.append(image_id)

    if not missing_ids:
        return

    for image_id in tqdm(missing_ids, desc="Captioning test images"):
        key = _caption_key(image_id, caption_model_name)
        idx = image_id_to_index[image_id]
        try:
            raw_image = _to_pil_image(dataset[idx]["image"])
            caption = caption_wrapper.caption_image(
                image_name=str(image_id),
                prompt=caption_prompt,
                raw_image=raw_image,
                temperature=caption_temperature,
                max_new_tokens=caption_max_new_tokens,
            )
            caption_cache[key] = _normalize_whitespace(caption) if caption else "unlabeled image"
        except Exception as exc:
            caption_cache[key] = f"unlabeled image ({exc.__class__.__name__})"


def _tokenize_for_model(tokenizer: Any, texts: list[str], device: str) -> Any:
    tokens = tokenizer(texts)
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output type: {type(tokens).__name__}")


def encode_text_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str,
) -> torch.Tensor:
    if not texts:
        raise ValueError("encode_text_batch received an empty text list.")
    normalized = [text if text else "unlabeled image" for text in texts]
    tokens = _tokenize_for_model(tokenizer, normalized, device)
    amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        if isinstance(tokens, dict):
            out = model.encode_text(**tokens)
        else:
            out = model.encode_text(tokens)
    emb = extract_text_embedding_tensor(out).float().cpu()
    emb /= emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return emb


def ensure_text_embeddings(
    *,
    keys: list[str],
    key_to_text: dict[str, str],
    emb_cache: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    missing = [key for key in keys if key not in emb_cache]
    for start in range(0, len(missing), batch_size):
        chunk_keys = missing[start : start + batch_size]
        chunk_texts = [key_to_text.get(key, "unlabeled image") for key in chunk_keys]
        chunk_emb = encode_text_batch(model, tokenizer, chunk_texts, device=device)
        for key, emb in zip(chunk_keys, chunk_emb):
            emb_cache[key] = emb
    return torch.stack([emb_cache[key] for key in keys])


def build_query_to_web_entries(
    metadata_records: list[dict],
    *,
    quality_config: WebImageQualityConfig,
    max_web_images_per_query: int | None,
    min_web_images_per_query: int | None,
    apply_filters: bool,
) -> tuple[dict[str, list[dict[str, str]]], list[dict], int, int]:
    query_to_entries: dict[str, list[dict[str, str]]] = {}
    filtered_records: list[dict] = []
    total_candidates = 0
    total_kept = 0

    for record in metadata_records:
        query_text = record.get("query")
        if not isinstance(query_text, str) or not query_text:
            continue

        original_paths = [p for p in record.get("image_paths", []) if isinstance(p, str)]
        image_urls = [u for u in record.get("image_urls", []) if isinstance(u, str)]
        total_candidates += len(original_paths)
        out_record = dict(record)
        entries: list[dict[str, str]] = []

        if apply_filters:
            kept_paths, accepted, rejected = filter_record_images(
                record,
                quality_config,
                max_keep=max_web_images_per_query,
                min_keep=min_web_images_per_query,
            )
            for item in accepted:
                path = str(item.get("path", ""))
                if not path:
                    continue
                text = _descriptor_from_fields(
                    path=path,
                    title=str(item.get("title", "")),
                    source=str(item.get("source", "")),
                    url=str(item.get("url", "")),
                    thumbnail=str(item.get("thumbnail", "")),
                )
                entries.append({"path": path, "text": text})

            out_record["image_paths_raw"] = original_paths
            out_record["image_paths"] = kept_paths
            out_record["image_quality_accepted"] = accepted
            out_record["image_quality_rejected"] = rejected
        else:
            if max_web_images_per_query is not None and max_web_images_per_query >= 0:
                final_paths = original_paths[:max_web_images_per_query]
            else:
                final_paths = original_paths

            by_url = _search_item_by_url(record)
            for idx, path in enumerate(final_paths):
                url = image_urls[idx] if idx < len(image_urls) else ""
                item = by_url.get(url, {})
                text = _descriptor_from_fields(
                    path=path,
                    title=str(item.get("title", "")) if isinstance(item, dict) else "",
                    source=str(item.get("source", "")) if isinstance(item, dict) else "",
                    url=url,
                    thumbnail=str(item.get("thumbnail", "")) if isinstance(item, dict) else "",
                )
                entries.append({"path": path, "text": text})
            out_record["image_paths"] = final_paths

        dedup_entries: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for entry in entries:
            path = entry.get("path", "")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            dedup_entries.append(entry)

        total_kept += len(dedup_entries)
        query_to_entries.setdefault(query_text, []).extend(dedup_entries)
        filtered_records.append(out_record)

    return query_to_entries, filtered_records, total_candidates, total_kept


def main() -> None:
    args = parse_args()
    patch_transformers_tokenizer_compat()

    split = args.split
    save_results_path = args.save_results_path or f"results_rerank_text_similarity_{split}.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.caption_cache_path:
        caption_cache_path = Path(args.caption_cache_path)
    else:
        caption_cache_path = Path(f"output/web_images/compare/caption_cache_{split}.json")

    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
    all_queries = dataset["query"]
    all_image_ids = dataset["inat24_image_id"]
    all_relevant = dataset["relevant"]
    unique_queries = np.unique(all_queries).tolist()
    if args.max_queries is not None:
        unique_queries = unique_queries[: args.max_queries]

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

    query_to_web_entries, filtered_records, total_candidates, total_kept = build_query_to_web_entries(
        metadata_records,
        quality_config=quality_config,
        max_web_images_per_query=args.max_web_images_per_query,
        min_web_images_per_query=args.min_web_images_per_query,
        apply_filters=args.apply_web_image_filters,
    )
    print(
        f"Web image candidates: {total_candidates}, kept: {total_kept}, "
        f"queries with web exemplars: {sum(1 for q in query_to_web_entries if query_to_web_entries[q])}"
    )

    if args.filtered_metadata_out:
        out_path = Path(args.filtered_metadata_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(filtered_records, f, indent=2, ensure_ascii=True)
        print(f"Saved filtered metadata to {out_path}")

    query_to_indices: dict[str, list[int]] = {}
    image_id_to_index: dict[Any, int] = {}
    for idx, (query, image_id) in enumerate(zip(all_queries, all_image_ids)):
        query_to_indices.setdefault(query, []).append(idx)
        if image_id not in image_id_to_index:
            image_id_to_index[image_id] = idx

    required_image_ids: list[Any] = []
    for query in unique_queries:
        if query_to_web_entries.get(query):
            required_image_ids.extend([all_image_ids[idx] for idx in query_to_indices.get(query, [])])

    caption_cache = {} if args.force_regenerate_captions else load_caption_cache(caption_cache_path)
    print(f"Caption cache entries loaded: {len(caption_cache)} from {caption_cache_path}")

    print(f"Loading caption VLM on {device}: {args.caption_vlm_model}")
    caption_wrapper = load_caption_model_with_retries(args, device=device)
    generate_missing_captions(
        image_ids=required_image_ids,
        caption_cache=caption_cache,
        force_regenerate=args.force_regenerate_captions,
        caption_model_name=args.caption_vlm_model,
        caption_wrapper=caption_wrapper,
        dataset=dataset,
        image_id_to_index=image_id_to_index,
        caption_prompt=args.caption_prompt,
        caption_max_new_tokens=args.caption_max_new_tokens,
        caption_temperature=args.caption_temperature,
    )
    save_caption_cache(caption_cache_path, caption_cache)
    print(f"Saved caption cache to {caption_cache_path} ({len(caption_cache)} entries)")

    if getattr(caption_wrapper, "model", None) is not None:
        del caption_wrapper.model
    if getattr(caption_wrapper, "processor", None) is not None:
        del caption_wrapper.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        selected_models = {name: all_models_available[name] for name in args.models}
    else:
        selected_models = all_models_available

    results: list[dict[str, Any]] = []
    skipped_models: list[tuple[str, str]] = []

    for title, clip_name in selected_models.items():
        if clip_name.startswith("timm:"):
            skipped_models.append((title, "no text encoder available"))
            print(f"[{title}] skipped: no text encoder available.")
            continue

        model = tokenizer = None
        last_exc: Exception | None = None
        for attempt in range(1, args.model_load_retries + 2):
            try:
                model, _, tokenizer = load_clip(clip_name, use_jit=False, device=device)
                if not hasattr(model, "encode_text"):
                    raise AttributeError("loaded model has no encode_text method")
                model.eval()
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
        if model is None or tokenizer is None:
            reason = f"{last_exc.__class__.__name__}: {last_exc}" if last_exc else "unknown load error"
            skipped_models.append((title, reason))
            print(f"[{title}] skipped after load failures: {reason}")
            continue

        text_emb_cache: dict[str, torch.Tensor] = {}
        metrics_avg = MetricAverage()
        evaluated_queries = 0

        for query in tqdm(unique_queries, desc=f"{title} text rerank"):
            query_indices = query_to_indices.get(query, [])
            if not query_indices:
                continue

            web_entries = query_to_web_entries.get(query, [])
            if not web_entries:
                continue
            web_entries = [entry for entry in web_entries if Path(entry.get("path", "")).exists()]
            if not web_entries:
                continue

            image_ids = [all_image_ids[idx] for idx in query_indices]
            candidate_keys = [_caption_key(image_id, args.caption_vlm_model) for image_id in image_ids]
            key_to_candidate_text = {key: caption_cache.get(key, "unlabeled image") for key in candidate_keys}
            candidate_embs = ensure_text_embeddings(
                keys=candidate_keys,
                key_to_text=key_to_candidate_text,
                emb_cache=text_emb_cache,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
            )

            web_keys = [f"web:{entry['path']}" for entry in web_entries]
            key_to_web_text = {f"web:{entry['path']}": entry.get("text", "unlabeled web image") for entry in web_entries}
            web_embs = ensure_text_embeddings(
                keys=web_keys,
                key_to_text=key_to_web_text,
                emb_cache=text_emb_cache,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
            )

            sim_matrix = candidate_embs.float() @ web_embs.float().T
            y_pred = aggregate_similarity_scores(
                sim_matrix,
                mode=args.aggregation,
                top_k=args.top_k_sims,
            ).numpy()
            y_true = np.asarray([all_relevant[idx] for idx in query_indices])

            pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
            metrics_avg.update([ap * 100, ndcg * 100, mrr])
            results.append(dict(model=title, query=query, ap=ap * 100, ndcg=ndcg * 100, mrr=mrr))
            evaluated_queries += 1

        if metrics_avg.avg is None:
            print(f"{title:30s}\tno valid queries after filtering")
        else:
            ap, ndcg, mrr = metrics_avg.avg
            print(f"{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}\tqueries={evaluated_queries}")

        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped_models:
        print("Skipped models:")
        for name, reason in skipped_models:
            print(f"- {name}: {reason}")

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
