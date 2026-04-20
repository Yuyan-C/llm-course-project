"""Evaluate a planner-based orchestrator with 4 retrieval options on INQUIRE-Rerank.

Supported retrieval modes:
- t2i: query text vs candidate image embeddings
- i2i: web image exemplars vs candidate image embeddings
- t2t: query text vs candidate dataset text (no LLM generation)
- t2t_augmented: query text vs augmented candidate captions (caption + metadata)
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inquire.lmm_utils_new import ModelWrapper
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.utils import load_clip
from src.inquire.web_image_quality import (
    WebImageQualityConfig,
    extract_image_embedding_tensor,
    filter_record_images,
)


LOGGER = logging.getLogger("eval_orchestrator")


MODE_ALIASES = {
    "t2i": "t2i",
    "text_to_image": "t2i",
    "text-to-image": "t2i",
    "i2i": "i2i",
    "image_to_image": "i2i",
    "image-to-image": "i2i",
    "t2t": "t2t",
    "text_to_text": "t2t",
    "text-to-text": "t2t",
    "t2t_augmented": "t2t_augmented",
    "t2t-augmented": "t2t_augmented",
    "t2t_aug": "t2t_augmented",
}

MODEL_KEY_TO_NAME = {
    "vit-b-32": "hf_clip:openai/clip-vit-base-patch32",
    "bioclip": "bioclip",
    "biocap": "biocap",
    "siglip-vit-b-16": "open_clip:ViT-B-16-SigLIP-256/webli",
    "siglip-so400m-14-384": "open_clip:ViT-SO400M-14-SigLIP-384/webli",
}


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_config(config_path: str) -> SimpleNamespace:
    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return to_namespace(payload)


def patch_transformers_tokenizer_compat() -> None:
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


def load_planner_model(model_name: str, retries: int = 3, sleep_s: float = 10.0) -> tuple[Any, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            # use_fast=False avoids optional protobuf/tokenizers paths that have
            # repeatedly failed in this environment.
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    dtype=torch.float32,
                    trust_remote_code=True,
                )
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token_id
            return model, tokenizer
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > retries:
                break
            print(
                f"[planner-model] load failed on attempt {attempt}/{retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {sleep_s:.1f}s..."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import time
            time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load planner model.")


def normalize_mode(value: str | None) -> str | None:
    if value is None:
        return None
    key = re.sub(r"\s+", "_", value.strip().lower())
    return MODE_ALIASES.get(key)


def _extract_first_json_block(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                block = text[start : idx + 1]
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_planner_decision(response_text: str) -> dict[str, Any]:
    payload = _extract_first_json_block(response_text)
    if payload is None:
        return {"retrieval_mode": None, "rewritten_query": None, "raw": response_text}

    retrieval_mode = normalize_mode(payload.get("retrieval_mode"))
    rewritten_query = payload.get("rewritten_query")
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        rewritten_query = None

    return {
        "retrieval_mode": retrieval_mode,
        "rewritten_query": rewritten_query,
        "payload": payload,
        "raw": response_text,
    }


def resolve_embedding_model(model_key_or_name: str) -> str:
    return MODEL_KEY_TO_NAME.get(model_key_or_name, model_key_or_name)


def _extract_text_embedding_tensor(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output

    for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
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

    raise TypeError(f"Unsupported text embedding output type: {type(model_output).__name__}")


def _tokenize_for_model(tokenizer: Any, texts: list[str], device: str) -> Any:
    tokens = tokenizer(texts)
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output type: {type(tokens).__name__}")


def encode_text_batch(model: Any, tokenizer: Any, texts: list[str], device: str) -> torch.Tensor:
    if not texts:
        raise ValueError("encode_text_batch received empty texts")
    normalized = [text if text else "unlabeled image" for text in texts]
    tokens = _tokenize_for_model(tokenizer, normalized, device)
    amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        if isinstance(tokens, dict):
            out = model.encode_text(**tokens)
        else:
            out = model.encode_text(tokens)
    emb = _extract_text_embedding_tensor(out).float().cpu()
    emb /= emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return emb


def ensure_text_embeddings(
    *,
    keys: list[str],
    key_to_text: dict[str, str],
    emb_cache: dict[str, torch.Tensor],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    missing = [key for key in keys if key not in emb_cache]
    for start in range(0, len(missing), max(1, batch_size)):
        chunk_keys = missing[start : start + batch_size]
        chunk_texts = [key_to_text.get(key, "unlabeled image") for key in chunk_keys]
        chunk_emb = encode_text_batch(model, tokenizer, chunk_texts, device)
        for key, emb in zip(chunk_keys, chunk_emb):
            emb_cache[key] = emb
    return torch.stack([emb_cache[key] for key in keys])


def load_caption_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def save_caption_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=True)


def caption_key(image_id: Any, caption_model: str) -> str:
    return f"{caption_model}::inat24:{image_id}"


def _to_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        return Image.fromarray(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type for captioning: {type(image_obj).__name__}")


def load_caption_model_with_retries(model_name: str, device: str, retries: int, sleep_s: float) -> ModelWrapper:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return ModelWrapper(model_name, device=device)
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > retries:
                break
            print(
                f"[caption-model] load failed on attempt {attempt}/{retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {sleep_s:.1f}s..."
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import time
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load caption model.")


def generate_missing_captions(
    *,
    image_ids: list[Any],
    caption_cache: dict[str, str],
    caption_model_name: str,
    caption_wrapper: ModelWrapper,
    dataset: Any,
    image_id_to_dataset_index: dict[Any, int],
    caption_prompt: str,
    caption_max_new_tokens: int,
    caption_temperature: float,
) -> int:
    missing = []
    for image_id in dict.fromkeys(image_ids):
        key = caption_key(image_id, caption_model_name)
        if caption_cache.get(key):
            continue
        if image_id in image_id_to_dataset_index:
            missing.append(image_id)

    if not missing:
        return 0

    for image_id in tqdm(missing, desc="Captioning"):
        key = caption_key(image_id, caption_model_name)
        idx = image_id_to_dataset_index[image_id]
        try:
            raw_image = _to_pil_image(dataset[idx]["image"])
            caption = caption_wrapper.caption_image(
                image_name=str(image_id),
                prompt=caption_prompt,
                raw_image=raw_image,
                temperature=caption_temperature,
                max_new_tokens=caption_max_new_tokens,
            )
            caption_cache[key] = " ".join(str(caption).split()) if caption else "unlabeled image"
        except Exception as exc:
            caption_cache[key] = f"unlabeled image ({exc.__class__.__name__})"
    return len(missing)


def normalize_metadata(records_obj: object) -> list[dict]:
    if isinstance(records_obj, list):
        return [item for item in records_obj if isinstance(item, dict)]
    if isinstance(records_obj, dict):
        return [records_obj]
    raise ValueError(f"Expected metadata JSON list/dict, got {type(records_obj).__name__}")


def build_query_to_downloads(
    metadata_records: list[dict],
    *,
    quality_config: WebImageQualityConfig,
    max_web_images_per_query: int | None,
    min_web_images_per_query: int | None,
    apply_filters: bool,
) -> dict[str, list[str]]:
    query_to_downloads: dict[str, list[str]] = {}

    for record in metadata_records:
        query_text = record.get("query")
        if not isinstance(query_text, str) or not query_text:
            continue

        original_paths = [p for p in record.get("image_paths", []) if isinstance(p, str)]

        if apply_filters:
            kept_paths, _, _ = filter_record_images(
                record,
                quality_config,
                max_keep=max_web_images_per_query,
                min_keep=min_web_images_per_query,
            )
            final_paths = kept_paths
        else:
            if max_web_images_per_query is not None and max_web_images_per_query >= 0:
                final_paths = original_paths[:max_web_images_per_query]
            else:
                final_paths = original_paths

        dedup = []
        seen: set[str] = set()
        for path in final_paths:
            if path in seen:
                continue
            seen.add(path)
            dedup.append(path)
        query_to_downloads[query_text] = dedup

    return query_to_downloads


def aggregate_similarity_scores(sim_matrix: torch.Tensor, mode: str, top_k: int) -> torch.Tensor:
    if sim_matrix.numel() == 0:
        return torch.empty((sim_matrix.shape[0],), dtype=sim_matrix.dtype)
    if mode == "mean":
        return sim_matrix.mean(dim=1)
    k = sim_matrix.shape[1] if top_k <= 0 else min(top_k, sim_matrix.shape[1])
    return sim_matrix.topk(k, dim=1).values.mean(dim=1)


def _normalize_text_field_names(raw_fields: Any, default_fields: list[str]) -> list[str]:
    if isinstance(raw_fields, str):
        fields = [raw_fields]
    elif isinstance(raw_fields, (list, tuple)):
        fields = [str(v) for v in raw_fields]
    else:
        fields = list(default_fields)

    valid = {"species", "category", "supercategory", "iconic_group", "query"}
    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        key = field.strip().lower()
        if key in valid and key not in seen:
            normalized.append(key)
            seen.add(key)
    return normalized if normalized else list(default_fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4-mode orchestrator on INQUIRE-Rerank.")
    parser.add_argument("--config", type=str, default="configs/eval_orchestrator.yml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-images-per-query", type=int, default=None)
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="auto",
        choices=["auto", "t2i", "i2i", "t2t", "t2t_augmented"],
        help="auto: use planner-selected mode; else force one mode.",
    )
    parser.add_argument(
        "--retrieval-model",
        type=str,
        default=None,
        help="Embedding model key/name for scoring (e.g., siglip-so400m-14-384).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--caption-cache-path", type=str, default=None)
    parser.add_argument("--save-results-path", type=str, default=None)
    parser.add_argument("--save-query-details-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    return parser.parse_args()


def run_planner_once(model: Any, tokenizer: Any, query_text: str, config: SimpleNamespace) -> str:
    messages = [
        {"role": "system", "content": config.init_prompt},
        {"role": "user", "content": query_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=config.chat_template.enable_thinking,
    ).to(model.device)

    max_new_tokens = int(config.generate.max_new_tokens)
    temperature = float(config.generate.decode.temperature)
    do_sample = bool(config.generate.decode.do_sample)
    top_p = float(config.generate.decode.top_p)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    patch_transformers_tokenizer_compat()

    config = load_config(args.config)
    if args.image_dir:
        config.image_dir = args.image_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    planner_model, planner_tokenizer = load_planner_model(config.model.name)

    split_name = "validation" if args.split == "val" else "test"
    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=split_name)

    all_queries = dataset["query"]
    all_relevant = dataset["relevant"]
    all_image_ids = dataset["inat24_image_id"]
    all_file_names = dataset["inat24_file_name"]
    all_species = dataset["inat24_species_name"]
    all_supercategory = dataset["supercategory"]
    all_category = dataset["category"]
    all_iconic_group = dataset["iconic_group"]

    query_to_indices: dict[str, list[int]] = {}
    image_id_to_dataset_index: dict[Any, int] = {}
    for idx, (query, image_id) in enumerate(zip(all_queries, all_image_ids)):
        query_to_indices.setdefault(query, []).append(idx)
        if image_id not in image_id_to_dataset_index:
            image_id_to_dataset_index[image_id] = idx

    unique_queries = np.unique(all_queries).tolist()
    if args.max_queries is not None:
        unique_queries = unique_queries[: args.max_queries]

    retrieval_model_name = (
        args.retrieval_model
        if args.retrieval_model
        else getattr(getattr(config, "retrieval", SimpleNamespace()), "embedding_model", "siglip-so400m-14-384")
    )
    clip_name = resolve_embedding_model(retrieval_model_name)
    retrieval_model, preprocess, retrieval_tokenizer = load_clip(clip_name, use_jit=False, device=device)
    if not hasattr(retrieval_model, "encode_image"):
        raise RuntimeError(f"Retrieval model does not expose encode_image: {clip_name}")

    # Precompute candidate image embeddings once.
    needed_image_ids: set[Any] = set()
    for q in unique_queries:
        for idx in query_to_indices.get(q, []):
            needed_image_ids.add(all_image_ids[idx])

    def collate_transform(examples):
        pixels = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex["inat24_image_id"] for ex in examples]
        return pixels, ids

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        collate_fn=collate_transform,
        num_workers=args.num_workers,
    )

    image_emb_cache: dict[Any, torch.Tensor] = {}
    amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
    for images, ids in tqdm(dataloader, desc="Embedding candidate images"):
        keep_mask = [img_id in needed_image_ids for img_id in ids]
        if not any(keep_mask):
            continue
        with torch.no_grad(), amp_ctx:
            embs = extract_image_embedding_tensor(retrieval_model.encode_image(images.to(device))).float().cpu()
            embs /= embs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        for img_id, emb, keep in zip(ids, embs, keep_mask):
            if keep:
                image_emb_cache[img_id] = emb

    # Shared caches for text embeddings and optional captioning.
    text_emb_cache: dict[str, torch.Tensor] = {}

    retrieval_cfg = getattr(config, "retrieval", SimpleNamespace())
    default_mode = normalize_mode(getattr(retrieval_cfg, "default_mode", "t2t")) or "t2t"
    use_rewritten_query = bool(getattr(retrieval_cfg, "use_rewritten_query", True))

    caption_model_name = getattr(retrieval_cfg, "caption_model", "Qwen/Qwen3-VL-4B-Instruct")
    caption_prompt = getattr(
        retrieval_cfg,
        "caption_prompt",
        "Describe this image in one concise factual sentence. Focus on species, appearance, behavior, and context.",
    )
    caption_max_new_tokens = int(getattr(retrieval_cfg, "caption_max_new_tokens", 48))
    caption_temperature = float(getattr(retrieval_cfg, "caption_temperature", 0.0))
    caption_load_retries = int(getattr(retrieval_cfg, "caption_load_retries", 3))
    caption_load_retry_sleep = float(getattr(retrieval_cfg, "caption_load_retry_sleep", 10.0))

    if args.caption_cache_path:
        caption_cache_path = Path(args.caption_cache_path)
    else:
        caption_cache_path = Path(
            getattr(
                retrieval_cfg,
                "caption_cache_path",
                f"output/web_images/compare/caption_cache_{args.split}.json",
            )
        )

    caption_cache = load_caption_cache(caption_cache_path)
    caption_wrapper: ModelWrapper | None = None

    i2i_cfg = getattr(retrieval_cfg, "i2i", SimpleNamespace())
    i2i_metadata_path = getattr(
        i2i_cfg,
        "metadata",
        "/network/scratch/y/yuyan.chen/inquire/web_images_raw_top20/image_search_metadata_test.json",
    )
    i2i_max_web = int(getattr(i2i_cfg, "max_web_images_per_query", 8))
    i2i_min_web = int(getattr(i2i_cfg, "min_web_images_per_query", 2))
    i2i_apply_filters = bool(getattr(i2i_cfg, "apply_web_image_filters", False))
    i2i_aggregation = str(getattr(i2i_cfg, "aggregation", "topk_mean"))
    i2i_top_k = int(getattr(i2i_cfg, "top_k", 3))
    i2i_missing_behavior = str(getattr(i2i_cfg, "missing_behavior", "fallback_t2i"))

    quality_config = WebImageQualityConfig(
        min_short_side=int(getattr(i2i_cfg, "min_short_side", 224)),
        max_aspect_ratio=float(getattr(i2i_cfg, "max_aspect_ratio", 2.8)),
        stock_thumbnail_short_side=int(getattr(i2i_cfg, "stock_thumbnail_short_side", 600)),
        metadata_relevance_threshold=float(getattr(i2i_cfg, "metadata_relevance_threshold", 0.25)),
        drop_stock_thumbnails=True,
    )

    query_to_web_paths: dict[str, list[str]] | None = None
    web_emb_cache: dict[str, torch.Tensor] = {}

    t2t_cfg = getattr(retrieval_cfg, "t2t", SimpleNamespace())
    t2t_fields = _normalize_text_field_names(getattr(t2t_cfg, "base_fields", ["species"]), ["species"])

    t2t_aug_cfg = getattr(retrieval_cfg, "t2t_augmented", SimpleNamespace())
    t2t_aug_fields = _normalize_text_field_names(
        getattr(t2t_aug_cfg, "metadata_fields", ["species", "category", "supercategory", "iconic_group"]),
        ["species", "category", "supercategory", "iconic_group"],
    )

    results: list[dict[str, Any]] = []
    details_dir = Path(args.save_query_details_dir) if args.save_query_details_dir else None
    if details_dir is not None:
        details_dir.mkdir(parents=True, exist_ok=True)

    metrics_avg = MetricAverage()

    def ensure_query_web_paths() -> dict[str, list[str]]:
        nonlocal query_to_web_paths
        if query_to_web_paths is not None:
            return query_to_web_paths
        with Path(i2i_metadata_path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        records = normalize_metadata(raw)
        query_to_web_paths = build_query_to_downloads(
            records,
            quality_config=quality_config,
            max_web_images_per_query=i2i_max_web,
            min_web_images_per_query=i2i_min_web,
            apply_filters=i2i_apply_filters,
        )
        return query_to_web_paths

    def embed_web_images(paths: list[str]) -> torch.Tensor:
        missing = [path for path in paths if path not in web_emb_cache]
        if missing:
            batch_images: list[torch.Tensor] = []
            batch_paths: list[str] = []
            for path in missing:
                p = Path(path)
                if not p.exists():
                    continue
                try:
                    with Image.open(p) as img:
                        batch_images.append(preprocess(img.convert("RGB")).unsqueeze(0))
                        batch_paths.append(path)
                except Exception:
                    continue
            if batch_images:
                batch = torch.cat(batch_images)
                with torch.no_grad(), amp_ctx:
                    emb = extract_image_embedding_tensor(retrieval_model.encode_image(batch.to(device))).float().cpu()
                    emb /= emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                for path, e in zip(batch_paths, emb):
                    web_emb_cache[path] = e

        existing = [web_emb_cache[path] for path in paths if path in web_emb_cache]
        if not existing:
            dim = next(iter(image_emb_cache.values())).shape[-1]
            return torch.empty((0, dim))
        return torch.stack(existing)

    for q_idx, query_text in enumerate(unique_queries):
        LOGGER.info("[%s/%s] Orchestrating query: %s", q_idx + 1, len(unique_queries), query_text)

        planner_response = run_planner_once(planner_model, planner_tokenizer, query_text, config)
        decision = parse_planner_decision(planner_response)
        planner_mode = decision.get("retrieval_mode")
        rewritten_query = decision.get("rewritten_query")

        if args.retrieval_mode == "auto":
            retrieval_mode = planner_mode or default_mode
        else:
            retrieval_mode = args.retrieval_mode

        if retrieval_mode not in {"t2i", "i2i", "t2t", "t2t_augmented"}:
            retrieval_mode = default_mode

        effective_query = rewritten_query if (use_rewritten_query and rewritten_query) else query_text

        query_indices = query_to_indices.get(query_text, [])
        if args.max_images_per_query is not None:
            query_indices = query_indices[: args.max_images_per_query]
        if not query_indices:
            continue

        image_ids = [all_image_ids[idx] for idx in query_indices]
        candidate_embs = torch.stack([image_emb_cache[image_id] for image_id in image_ids])
        y_true = np.asarray([all_relevant[idx] for idx in query_indices])
        count_pos = int(y_true.sum())
        if count_pos == 0:
            continue

        y_pred: np.ndarray
        mode_notes: dict[str, Any] = {}

        if retrieval_mode == "t2i":
            q_key = f"query::{effective_query}"
            q_emb = ensure_text_embeddings(
                keys=[q_key],
                key_to_text={q_key: effective_query},
                emb_cache=text_emb_cache,
                model=retrieval_model,
                tokenizer=retrieval_tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )[0]
            y_pred = (candidate_embs.float() @ q_emb.float()).numpy()

        elif retrieval_mode == "i2i":
            query_web = ensure_query_web_paths().get(query_text, [])
            query_web = [path for path in query_web if Path(path).exists()]
            web_embs = embed_web_images(query_web)
            if web_embs.numel() == 0:
                mode_notes["i2i_missing_web_images"] = True
                if i2i_missing_behavior == "zeros":
                    y_pred = np.zeros((len(query_indices),), dtype=np.float32)
                else:
                    # fallback to t2i
                    q_key = f"query::{effective_query}"
                    q_emb = ensure_text_embeddings(
                        keys=[q_key],
                        key_to_text={q_key: effective_query},
                        emb_cache=text_emb_cache,
                        model=retrieval_model,
                        tokenizer=retrieval_tokenizer,
                        device=device,
                        batch_size=max(1, args.batch_size),
                    )[0]
                    y_pred = (candidate_embs.float() @ q_emb.float()).numpy()
                    mode_notes["i2i_fallback"] = "t2i"
            else:
                sim_matrix = candidate_embs.float() @ web_embs.float().T
                y_pred = aggregate_similarity_scores(sim_matrix, mode=i2i_aggregation, top_k=i2i_top_k).numpy()
                mode_notes["n_web_images"] = int(web_embs.shape[0])

        else:
            # t2t and t2t_augmented
            candidate_texts: dict[str, str] = {}
            candidate_keys: list[str] = []

            if retrieval_mode == "t2t":
                for idx, image_id in zip(query_indices, image_ids):
                    key = f"t2t_base::{image_id}"
                    candidate_keys.append(key)
                    parts: list[str] = []
                    for field in t2t_fields:
                        if field == "species":
                            value = all_species[idx]
                        elif field == "category":
                            value = all_category[idx]
                        elif field == "supercategory":
                            value = all_supercategory[idx]
                        elif field == "iconic_group":
                            value = all_iconic_group[idx]
                        else:
                            value = query_text
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                    candidate_texts[key] = " ; ".join(parts) if parts else "unlabeled image"
                mode_notes["t2t_fields"] = t2t_fields
            else:
                if caption_wrapper is None:
                    caption_wrapper = load_caption_model_with_retries(
                        caption_model_name,
                        device=device,
                        retries=caption_load_retries,
                        sleep_s=caption_load_retry_sleep,
                    )

                generated = generate_missing_captions(
                    image_ids=image_ids,
                    caption_cache=caption_cache,
                    caption_model_name=caption_model_name,
                    caption_wrapper=caption_wrapper,
                    dataset=dataset,
                    image_id_to_dataset_index=image_id_to_dataset_index,
                    caption_prompt=caption_prompt,
                    caption_max_new_tokens=caption_max_new_tokens,
                    caption_temperature=caption_temperature,
                )
                if generated:
                    save_caption_cache(caption_cache_path, caption_cache)

                for idx, image_id in zip(query_indices, image_ids):
                    key = caption_key(image_id, caption_model_name)
                    candidate_keys.append(key)
                    caption = caption_cache.get(key, "unlabeled image")
                    parts = [caption]
                    for field in t2t_aug_fields:
                        if field == "species":
                            parts.append(f"Species: {all_species[idx]}.")
                        elif field == "category":
                            parts.append(f"Category: {all_category[idx]}.")
                        elif field == "supercategory":
                            parts.append(f"Supercategory: {all_supercategory[idx]}.")
                        elif field == "iconic_group":
                            parts.append(f"Iconic group: {all_iconic_group[idx]}.")
                        elif field == "query":
                            parts.append(f"Original query: {query_text}.")
                    candidate_texts[key] = " ".join(parts)
                mode_notes["t2t_augmented_fields"] = t2t_aug_fields

            query_for_text = effective_query
            if retrieval_mode == "t2t_augmented" and rewritten_query and rewritten_query != query_text:
                query_for_text = f"{query_text}. {rewritten_query}"

            q_key = f"query::{query_for_text}"
            q_emb = ensure_text_embeddings(
                keys=[q_key],
                key_to_text={q_key: query_for_text},
                emb_cache=text_emb_cache,
                model=retrieval_model,
                tokenizer=retrieval_tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )[0]
            c_embs = ensure_text_embeddings(
                keys=candidate_keys,
                key_to_text=candidate_texts,
                emb_cache=text_emb_cache,
                model=retrieval_model,
                tokenizer=retrieval_tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )
            y_pred = (c_embs.float() @ q_emb.float()).numpy()

        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=count_pos)
        metrics_avg.update([ap * 100, ndcg * 100, mrr])

        results.append(
            {
                "model": f"orchestrator::{config.model.name}",
                "query": query_text,
                "planner_mode": planner_mode,
                "retrieval_mode": retrieval_mode,
                "used_rewritten_query": int(bool(use_rewritten_query and rewritten_query)),
                "effective_query": effective_query,
                "ap": ap * 100,
                "ndcg": ndcg * 100,
                "mrr": mrr,
                "precision": pr,
                "recall": rec,
                "n_images": len(query_indices),
                "n_pos": count_pos,
                "mode_notes": json.dumps(mode_notes, ensure_ascii=True),
            }
        )

        if details_dir is not None:
            details = {
                "query": query_text,
                "planner_mode": planner_mode,
                "retrieval_mode": retrieval_mode,
                "rewritten_query": rewritten_query,
                "effective_query": effective_query,
                "count_pos": count_pos,
                "mode_notes": mode_notes,
                "planner_payload": decision.get("payload"),
                "planner_raw": decision.get("raw"),
            }
            with (details_dir / f"query_{q_idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(details, f, indent=2, ensure_ascii=True)

    if caption_wrapper is not None:
        if getattr(caption_wrapper, "model", None) is not None:
            del caption_wrapper.model
        if getattr(caption_wrapper, "processor", None) is not None:
            del caption_wrapper.processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_df = pd.DataFrame.from_dict(results)
    if len(results_df) == 0:
        raise RuntimeError("No query-level rows were produced.")

    if args.save_results_path:
        output_path = Path(args.save_results_path)
    else:
        output_path = Path(f"output/web_images/compare/results_orchestrator_{args.split}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    pd.options.display.float_format = " {:,.2f}".format
    print("Overall summary:")
    print(results_df.groupby("model").agg({"ap": "mean", "ndcg": "mean", "mrr": "mean"}).sort_values("ap"))
    print("\nBy retrieval_mode:")
    print(results_df.groupby("retrieval_mode").agg({"ap": "mean", "ndcg": "mean", "mrr": "mean", "query": "count"}))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
