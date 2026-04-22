"""Evaluate text-space reranking using test-image captions and query text."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from PIL import Image

from src.inquire.lmm_utils_new import ModelWrapper
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.utils import load_clip


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
    parser = argparse.ArgumentParser(
        description="Run query-text to caption-text reranking on INQUIRE-Rerank.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--save-results-path", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of retrieval model keys to evaluate (e.g. siglip-so400m-14-384).",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-load-retries", type=int, default=3)
    parser.add_argument("--model-load-retry-sleep", type=float, default=5.0)

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
    parser.add_argument(
        "--caption-only",
        action="store_true",
        help="Generate/update caption cache and exit without running text reranking.",
    )
    parser.add_argument(
        "--caption-num-shards",
        type=int,
        default=1,
        help="Split caption generation into N deterministic shards.",
    )
    parser.add_argument(
        "--caption-shard-index",
        type=int,
        default=0,
        help="0-based shard index to run when --caption-num-shards > 1.",
    )
    parser.add_argument(
        "--caption-save-every",
        type=int,
        default=200,
        help="Persist caption cache every N generated captions (0 disables periodic saves).",
    )
    return parser.parse_args()


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


def select_image_ids_for_shard(
    image_ids: list[Any],
    *,
    num_shards: int,
    shard_index: int,
) -> list[Any]:
    if num_shards < 1:
        raise ValueError("--caption-num-shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--caption-shard-index must be in [0, {num_shards - 1}] when --caption-num-shards={num_shards}.",
        )
    unique_image_ids = list(dict.fromkeys(image_ids))
    if num_shards == 1:
        return unique_image_ids
    return [image_id for idx, image_id in enumerate(unique_image_ids) if idx % num_shards == shard_index]


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
    caption_cache_path: Path,
    caption_save_every: int,
    progress_desc: str,
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

    for processed, image_id in enumerate(tqdm(missing_ids, desc=progress_desc), start=1):
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
            caption_cache[key] = " ".join(str(caption).split()) if caption else "unlabeled image"
        except Exception as exc:
            caption_cache[key] = f"unlabeled image ({exc.__class__.__name__})"
        if caption_save_every > 0 and (processed % caption_save_every == 0):
            save_caption_cache(caption_cache_path, caption_cache)


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


def main() -> None:
    args = parse_args()
    patch_transformers_tokenizer_compat()

    if args.caption_num_shards < 1:
        raise ValueError("--caption-num-shards must be >= 1.")
    if args.caption_shard_index < 0 or args.caption_shard_index >= args.caption_num_shards:
        raise ValueError(
            f"--caption-shard-index must be in [0, {args.caption_num_shards - 1}] "
            f"when --caption-num-shards={args.caption_num_shards}.",
        )

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

    query_to_indices: dict[str, list[int]] = {}
    image_id_to_index: dict[Any, int] = {}
    for idx, (query, image_id) in enumerate(zip(all_queries, all_image_ids)):
        query_to_indices.setdefault(query, []).append(idx)
        if image_id not in image_id_to_index:
            image_id_to_index[image_id] = idx

    required_image_ids = [
        all_image_ids[idx]
        for query in unique_queries
        for idx in query_to_indices.get(query, [])
    ]
    shard_image_ids = select_image_ids_for_shard(
        required_image_ids,
        num_shards=args.caption_num_shards,
        shard_index=args.caption_shard_index,
    )
    print(
        "Caption shard selection: "
        f"{len(shard_image_ids)} unique images in shard "
        f"{args.caption_shard_index + 1}/{args.caption_num_shards}."
    )

    caption_cache = {} if args.force_regenerate_captions else load_caption_cache(caption_cache_path)
    print(f"Caption cache entries loaded: {len(caption_cache)} from {caption_cache_path}")

    print(f"Loading caption VLM on {device}: {args.caption_vlm_model}")
    caption_wrapper = load_caption_model_with_retries(args, device=device)
    generate_missing_captions(
        image_ids=shard_image_ids,
        caption_cache=caption_cache,
        force_regenerate=args.force_regenerate_captions,
        caption_model_name=args.caption_vlm_model,
        caption_wrapper=caption_wrapper,
        dataset=dataset,
        image_id_to_index=image_id_to_index,
        caption_prompt=args.caption_prompt,
        caption_max_new_tokens=args.caption_max_new_tokens,
        caption_temperature=args.caption_temperature,
        caption_cache_path=caption_cache_path,
        caption_save_every=max(0, int(args.caption_save_every)),
        progress_desc=(
            f"Captioning {split} images (shard {args.caption_shard_index + 1}/{args.caption_num_shards})"
            if args.caption_num_shards > 1
            else f"Captioning {split} images"
        ),
    )
    save_caption_cache(caption_cache_path, caption_cache)
    print(f"Saved caption cache to {caption_cache_path} ({len(caption_cache)} entries)")

    if getattr(caption_wrapper, "model", None) is not None:
        del caption_wrapper.model
    if getattr(caption_wrapper, "processor", None) is not None:
        del caption_wrapper.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.caption_only:
        print("Caption-only mode enabled; skipping text reranking.")
        return

    all_models_available = {
        "vit-b-32": "hf_clip:openai/clip-vit-base-patch32",
        "bioclip": "bioclip",
        "biocap": "biocap",
        "siglip-vit-b-16": "open_clip:ViT-B-16-SigLIP-256/webli",
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

            query_key = f"query::{query}"
            query_emb = ensure_text_embeddings(
                keys=[query_key],
                key_to_text={query_key: query},
                emb_cache=text_emb_cache,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )[0]

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
                batch_size=max(1, args.batch_size),
            )

            y_pred = (candidate_embs.float() @ query_emb.float()).numpy()
            y_true = np.asarray([all_relevant[idx] for idx in query_indices])

            pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
            metrics_avg.update([ap * 100, ndcg * 100, mrr])
            results.append(dict(model=title, query=query, ap=ap * 100, ndcg=ndcg * 100, mrr=mrr))
            evaluated_queries += 1

        if metrics_avg.avg is None:
            print(f"{title:30s}\tno valid queries")
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
        print("No evaluation rows were produced.")
    else:
        print(results_df.groupby("model").agg({"ap": "mean", "ndcg": "mean", "mrr": "mean"}).sort_values("ap"))

    output_path = Path(save_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print("All done! Saved results to", output_path)


if __name__ == "__main__":
    main()
