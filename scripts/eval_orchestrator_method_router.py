"""Route each query to one method (i2i/t2i/t2t) using SmolLM and evaluate retrieval."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any

from datasets import load_dataset
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.inquire.lmm_utils_new import ModelWrapper
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.utils import load_clip
from src.inquire.web_image_quality import (
    WebImageQualityConfig,
    extract_image_embedding_tensor,
    filter_record_images,
)


ALLOWED_METHODS = ("i2i", "t2i", "t2t")
MODEL_REGISTRY = {
    "vit-b-32": "hf_clip:openai/clip-vit-base-patch32",
    "bioclip": "bioclip",
    "biocap": "biocap",
    "siglip-vit-b-16": "open_clip:ViT-B-16-SigLIP-256/webli",
}
DEFAULT_ROUTER_PROMPT = """
You are a retrieval method router.

Task:
- Choose exactly one method for this query: "i2i", "t2i", or "t2t".
- Use only query text semantics. Do not assume any runtime stats or exemplar availability.

Method definitions:
- i2i: compare candidate images against downloaded web exemplar images.
- t2i: compare query text embedding against candidate image embeddings.
- t2t: compare query text embedding against generated caption embeddings.

Routing rubric:
- Choose i2i for explicit visual morphology and fine-grained appearance cues:
  colors, patterns, markings, textures, shape/proportions, body-part traits,
  pose/viewpoint, or "looks like"/"similar appearance" style wording.
- Choose t2t for behavior/context/compositional semantics:
  actions, interactions, scene context, habitat/environment, temporal states,
  relations between entities, or multi-clause natural-language descriptions.
- Choose t2i for concise object/species-centric naming queries:
  short noun phrases, taxonomic labels, or simple class-level mentions with
  little or no behavioral/contextual detail.

Tie-breakers:
- If behavior/context and morphology are both present, prefer t2t when behavior/context is central.
- If appearance detail is dominant and specific, prefer i2i.
- If the query is short and underspecified, prefer t2i.
- Never choose a method outside {"i2i","t2i","t2t"}.
- If i2i is chosen but exemplars are not usable at runtime, the system will fallback to t2i.

Return JSON only with this schema (placeholders, not literal values):
{"method":"<i2i|t2i|t2t>","confidence":<0_to_1_float>,"reason":"<query-specific rationale>","signals":["<signal_a>","<signal_b>"]}

Rules:
- Output exactly one JSON object and nothing else (no markdown, no prose outside JSON).
- "confidence" must be a numeric float between 0 and 1.
- "signals" must contain exactly two short, query-grounded signals.
- Do not copy placeholder tokens like <...>.
- Do not output generic text like "short rationale", "signal1", or "signal2".
- Make reason/signals specific to the given query.
""".strip()


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
    parser = argparse.ArgumentParser(description="SmolLM-routed retrieval over i2i/t2i/t2t.")
    parser.add_argument("--config", type=str, default="configs/eval.yml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Fixed retrieval model used by i2i/t2i/t2t for this run.",
    )
    parser.add_argument("--max-queries", type=int, default=None)

    parser.add_argument(
        "--metadata",
        type=str,
        default="/network/scratch/y/yuyan.chen/inquire/web_images/image_search_metadata_test.json",
        help="Web-image metadata JSON (defaults to shared scratch metadata).",
    )
    parser.add_argument("--max-web-images-per-query", type=int, default=8)
    parser.add_argument("--min-web-images-per-query", type=int, default=2)
    parser.add_argument("--disable-web-image-filters", action="store_true")
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.8)
    parser.add_argument("--stock-thumbnail-short-side", type=int, default=600)
    parser.add_argument("--metadata-relevance-threshold", type=float, default=0.25)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "topk_mean"])
    parser.add_argument("--top-k-sims", type=int, default=3)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-load-retries", type=int, default=3)
    parser.add_argument("--model-load-retry-sleep", type=float, default=5.0)

    parser.add_argument(
        "--caption-vlm-model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
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
    parser.add_argument("--caption-cache-path", type=str, default=None)
    parser.add_argument("--force-regenerate-captions", action="store_true")

    parser.add_argument("--orchestrator-model", type=str, default=None)
    parser.add_argument("--orchestrator-max-new-tokens", type=int, default=128)
    parser.add_argument("--orchestrator-temperature", type=float, default=0.0)
    parser.add_argument("--orchestrator-load-retries", type=int, default=2)
    parser.add_argument("--orchestrator-load-retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--fallback-method",
        type=str,
        default=None,
        choices=list(ALLOWED_METHODS),
    )
    parser.add_argument("--decision-cache-path", type=str, default=None)
    parser.add_argument("--force-reroute", action="store_true")

    parser.add_argument("--save-results-path", type=str, default=None)
    parser.add_argument("--save-summary-path", type=str, default=None)
    parser.add_argument("--save-usage-path", type=str, default=None)
    return parser.parse_args()


def load_yaml_config(path: str) -> SimpleNamespace:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    def to_ns(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in value.items()})
        if isinstance(value, list):
            return [to_ns(v) for v in value]
        return value

    return to_ns(payload)


def normalize_metadata(records_obj: object) -> list[dict[str, Any]]:
    if isinstance(records_obj, list):
        return [item for item in records_obj if isinstance(item, dict)]
    if isinstance(records_obj, dict):
        return [records_obj]
    raise ValueError(f"Expected metadata JSON to be a list or dict, got {type(records_obj).__name__}")


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
        f"Unsupported embedding output type for '{primary_attr}': {type(model_output).__name__}",
    )


def extract_text_embedding_tensor(model_output: Any) -> torch.Tensor:
    return _extract_embedding_tensor(model_output, "text_embeds")


def _tokenize_for_model(tokenizer: Any, texts: str | list[str], device: str) -> Any:
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


def _to_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        return Image.fromarray(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type for captioning: {type(image_obj).__name__}")


def _caption_key(image_id: Any, caption_model: str) -> str:
    return f"{caption_model}::inat24:{image_id}"


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
                f"{exc.__class__.__name__}: {exc}. Retrying in {args.caption_load_retry_sleep:.1f}s...",
            )
            time.sleep(args.caption_load_retry_sleep)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load caption model with unknown error.")


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
    topk_vals = sim_matrix.topk(k, dim=1).values
    return topk_vals.mean(dim=1)


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        objs.append(payload)
                    break
    return objs


def _normalize_method_candidate(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    method_raw = value.strip().lower().strip("`'\".,:; ")
    if method_raw in ALLOWED_METHODS:
        return method_raw

    # Accept minor formatting noise only if it maps to a single unambiguous method.
    mentions = re.findall(r"\b(i2i|t2i|t2t)\b", method_raw)
    uniq = sorted(set(mentions))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _parse_method_decision(raw_text: str, fallback_method: str) -> tuple[str, float | None, str, list[str], bool]:
    payloads = _extract_json_objects(raw_text)
    saw_json = bool(payloads)
    invalid_methods: list[str] = []

    for payload in payloads:
        method = _normalize_method_candidate(payload.get("method"))
        if method is None:
            invalid_methods.append(str(payload.get("method", "")))
            continue

        confidence = payload.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except Exception:
            confidence_value = None

        reason = str(payload.get("reason", "")).strip()
        raw_signals = payload.get("signals", [])
        signals = [str(item) for item in raw_signals] if isinstance(raw_signals, list) else []
        return method, confidence_value, reason, signals, False

    if saw_json:
        if invalid_methods:
            return fallback_method, None, f"invalid_method:{invalid_methods[0]}", [], True
        return fallback_method, None, "invalid_json", [], True

    text_mentions = re.findall(r"\b(i2i|t2i|t2t)\b", raw_text.lower())
    uniq_mentions = sorted(set(text_mentions))
    if len(uniq_mentions) == 1:
        return uniq_mentions[0], None, "parsed_from_text", [], False

    return fallback_method, None, "invalid_json", [], True


def _looks_like_placeholder_decision(reason: str, signals: list[str]) -> bool:
    reason_norm = re.sub(r"\s+", " ", str(reason).strip().lower())
    signal_norm = [re.sub(r"\s+", " ", str(item).strip().lower()) for item in signals]

    if reason_norm in {"short rationale", "rationale", "placeholder"}:
        return True
    if signal_norm and all(item in {"signal1", "signal2", "signal 1", "signal 2", "signal", "placeholder"} for item in signal_norm):
        return True
    return False


def _get_device_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def load_orchestrator_with_retries(
    *,
    model_name: str,
    device: str,
    retries: int,
    sleep_s: float,
) -> tuple[Any, Any]:
    last_exc: Exception | None = None
    dtype = _get_device_dtype(device)
    for attempt in range(1, retries + 2):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > retries:
                break
            print(
                f"[orchestrator] load failed on attempt {attempt}/{retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {sleep_s:.1f}s...",
            )
            time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load orchestrator with unknown error.")


def _build_router_input(
    *,
    query: str,
    query_indices: list[int],
) -> dict[str, Any]:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    latin_like = bool(re.search(r"\b[A-Z][a-z]{2,}\s+[a-z]{2,}\b", query))
    behavior_keywords = (
        "feeding",
        "foraging",
        "hunting",
        "nest",
        "nesting",
        "gathering",
        "flying",
        "perched",
        "swimming",
        "running",
        "camouflage",
    )
    has_behavior_words = any(word in query.lower() for word in behavior_keywords)

    return {
        "query": query,
        "query_stats": {
            "num_tokens": len(query_tokens),
            "num_chars": len(query),
            "looks_like_scientific_name": latin_like,
            "has_behavior_words": has_behavior_words,
            "num_candidate_images": len(query_indices),
        },
    }


def route_query_method(
    *,
    model: Any,
    tokenizer: Any,
    router_prompt: str,
    payload: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    fallback_method: str,
) -> dict[str, Any]:
    user_prompt = json.dumps(payload, ensure_ascii=True)
    messages = [
        {"role": "system", "content": router_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _generate(messages_in: list[dict[str, str]]) -> str:
        try:
            inputs = tokenizer.apply_chat_template(
                messages_in,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(model.device)
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages_in,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    raw_text = _generate(messages)
    method, confidence, reason, signals, used_fallback = _parse_method_decision(raw_text, fallback_method)

    if _looks_like_placeholder_decision(reason, signals):
        retry_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {
                "role": "user",
                "content": (
                    "You copied placeholder/example text. Re-evaluate this query and return concrete JSON only. "
                    "Do not use 'short rationale', 'signal1', or 'signal2'."
                ),
            },
        ]
        retry_raw_text = _generate(retry_messages)
        retry_method, retry_confidence, retry_reason, retry_signals, retry_used_fallback = _parse_method_decision(
            retry_raw_text,
            fallback_method,
        )
        if not _looks_like_placeholder_decision(retry_reason, retry_signals):
            raw_text = retry_raw_text
            method = retry_method
            confidence = retry_confidence
            reason = retry_reason
            signals = retry_signals
            used_fallback = retry_used_fallback
        else:
            used_fallback = True
            if reason:
                reason = f"{reason};placeholder_output"
            else:
                reason = "placeholder_output"

    return {
        "method": method,
        "confidence": confidence,
        "reason": reason,
        "signals": signals,
        "raw": raw_text,
        "invalid_decision": used_fallback,
    }


def load_retrieval_model_with_retries(
    *,
    model_key: str,
    device: str,
    retries: int,
    sleep_s: float,
) -> tuple[torch.nn.Module, Any, Any]:
    clip_name = MODEL_REGISTRY[model_key]
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            model, preprocess, tokenizer = load_clip(clip_name, use_jit=False, device=device)
            model.eval()
            return model, preprocess, tokenizer
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > retries:
                break
            print(
                f"[retrieval:{model_key}] load failed on attempt {attempt}/{retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {sleep_s:.1f}s...",
            )
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load retrieval model with unknown error.")


def build_query_to_downloads(
    metadata_records: list[dict[str, Any]],
    *,
    quality_config: WebImageQualityConfig,
    max_web_images_per_query: int | None,
    min_web_images_per_query: int | None,
    apply_filters: bool,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    query_to_downloads: dict[str, list[str]] = {}
    query_to_meta: dict[str, dict[str, Any]] = {}

    for record in metadata_records:
        query_text = record.get("query")
        if not isinstance(query_text, str) or not query_text:
            continue

        original_paths = [p for p in record.get("image_paths", []) if isinstance(p, str)]
        out_paths: list[str]
        accepted: list[dict[str, Any]] = []

        if apply_filters:
            out_paths, accepted, _ = filter_record_images(
                record,
                quality_config,
                max_keep=max_web_images_per_query,
                min_keep=min_web_images_per_query,
            )
        else:
            if max_web_images_per_query is not None and max_web_images_per_query >= 0:
                out_paths = original_paths[:max_web_images_per_query]
            else:
                out_paths = original_paths

        dedup_paths: list[str] = []
        seen: set[str] = set()
        for path in out_paths:
            if path in seen:
                continue
            seen.add(path)
            if Path(path).exists():
                dedup_paths.append(path)

        query_to_downloads[query_text] = dedup_paths

        short_sides = [float(item.get("short_side", 0.0)) for item in accepted if item.get("short_side") is not None]
        metadata_rel = [
            float(item.get("metadata_relevance", 0.0))
            for item in accepted
            if item.get("metadata_relevance") is not None
        ]
        query_to_meta[query_text] = {
            "num_web_candidates": len(original_paths),
            "num_web_images": len(dedup_paths),
            "num_web_accepted": len(accepted),
            "avg_short_side": float(np.mean(short_sides)) if short_sides else 0.0,
            "avg_metadata_relevance": float(np.mean(metadata_rel)) if metadata_rel else 0.0,
        }

    return query_to_downloads, query_to_meta


def load_decision_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Decision cache must be dict JSON: {path}")
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            result[key] = value
    return result


def save_decision_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    patch_transformers_tokenizer_compat()

    cfg = load_yaml_config(args.config)
    split = args.split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device={device}")

    orchestrator_cfg = getattr(cfg, "method_orchestrator", None)
    orchestrator_model_name = args.orchestrator_model
    if not orchestrator_model_name and orchestrator_cfg is not None:
        orchestrator_model_name = getattr(orchestrator_cfg, "model_name", None)
    if not orchestrator_model_name:
        orchestrator_model_name = "HuggingFaceTB/SmolLM3-3B"

    router_prompt = DEFAULT_ROUTER_PROMPT
    if orchestrator_cfg is not None and getattr(orchestrator_cfg, "router_prompt", None):
        router_prompt = str(orchestrator_cfg.router_prompt)
    fallback_method = args.fallback_method
    if fallback_method is None and orchestrator_cfg is not None:
        fallback_method = getattr(orchestrator_cfg, "fallback_method", None)
    if fallback_method is None:
        fallback_method = "t2i"

    if args.caption_cache_path:
        caption_cache_path = Path(args.caption_cache_path)
    else:
        caption_cache_path = Path(f"output/retrieval/method_router/caption_cache_{split}.json")

    compare_dir = Path("output/retrieval/method_router")
    compare_dir.mkdir(parents=True, exist_ok=True)
    default_results = compare_dir / f"results_orchestrator_{args.model_key}_{split}.csv"
    default_summary = compare_dir / f"results_orchestrator_{args.model_key}_{split}_summary.csv"
    default_usage = compare_dir / f"results_orchestrator_{args.model_key}_{split}_usage.csv"
    default_decision_cache = compare_dir / f"decision_cache_{args.model_key}_{split}.json"

    save_results_path = Path(args.save_results_path) if args.save_results_path else default_results
    save_summary_path = Path(args.save_summary_path) if args.save_summary_path else default_summary
    save_usage_path = Path(args.save_usage_path) if args.save_usage_path else default_usage
    decision_cache_path = Path(args.decision_cache_path) if args.decision_cache_path else default_decision_cache

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
    query_to_downloads, query_to_web_meta = build_query_to_downloads(
        metadata_records,
        quality_config=quality_config,
        max_web_images_per_query=args.max_web_images_per_query,
        min_web_images_per_query=args.min_web_images_per_query,
        apply_filters=not args.disable_web_image_filters,
    )
    n_with_web = sum(1 for q in unique_queries if query_to_downloads.get(q))
    print(f"Queries with usable web images: {n_with_web}/{len(unique_queries)}")

    retrieval_model, preprocess, retrieval_tokenizer = load_retrieval_model_with_retries(
        model_key=args.model_key,
        device=device,
        retries=args.model_load_retries,
        sleep_s=args.model_load_retry_sleep,
    )

    def collate_transform(examples):
        pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex["inat24_image_id"] for ex in examples]
        return pixel_values, ids

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_transform,
        num_workers=args.num_workers,
    )
    image_emb_cache: dict[Any, torch.Tensor] = {}
    for images, ids in tqdm(dataloader, total=max(1, len(dataset) // max(1, args.batch_size)), desc="Encoding dataset images"):
        amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            image_embs = extract_image_embedding_tensor(retrieval_model.encode_image(images.to(device))).float().cpu()
            image_embs /= image_embs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        image_emb_cache.update(dict(zip(ids, image_embs)))

    web_emb_cache: dict[str, torch.Tensor] = {}

    def get_web_embeddings(query_text: str) -> torch.Tensor:
        if query_text in web_emb_cache:
            return web_emb_cache[query_text]
        paths = query_to_downloads.get(query_text, [])
        tensors = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    tensors.append(preprocess(img.convert("RGB")).unsqueeze(0))
            except Exception:
                continue
        if not tensors:
            example_emb = next(iter(image_emb_cache.values()))
            empty = torch.empty((0, example_emb.shape[-1]))
            web_emb_cache[query_text] = empty
            return empty
        batch = torch.cat(tensors)
        amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            emb = extract_image_embedding_tensor(retrieval_model.encode_image(batch.to(device))).float().cpu()
            emb /= emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        web_emb_cache[query_text] = emb
        return emb

    caption_cache = {} if args.force_regenerate_captions else load_caption_cache(caption_cache_path)
    caption_wrapper: ModelWrapper | None = None
    text_emb_cache: dict[str, torch.Tensor] = {}

    def ensure_captions_for_query(query_text: str) -> None:
        nonlocal caption_wrapper
        query_indices = query_to_indices.get(query_text, [])
        image_ids = [all_image_ids[idx] for idx in query_indices]
        missing_ids: list[Any] = []
        for image_id in dict.fromkeys(image_ids):
            key = _caption_key(image_id, args.caption_vlm_model)
            if key in caption_cache and caption_cache[key] and not args.force_regenerate_captions:
                continue
            if image_id in image_id_to_index:
                missing_ids.append(image_id)
        if not missing_ids:
            return

        if caption_wrapper is None:
            caption_wrapper = load_caption_model_with_retries(args, device=device)

        for image_id in missing_ids:
            key = _caption_key(image_id, args.caption_vlm_model)
            idx = image_id_to_index[image_id]
            try:
                raw_image = _to_pil_image(dataset[idx]["image"])
                caption = caption_wrapper.caption_image(
                    image_name=str(image_id),
                    prompt=args.caption_prompt,
                    raw_image=raw_image,
                    temperature=args.caption_temperature,
                    max_new_tokens=args.caption_max_new_tokens,
                )
                caption_cache[key] = " ".join(str(caption).split()) if caption else "unlabeled image"
            except Exception as exc:
                caption_cache[key] = f"unlabeled image ({exc.__class__.__name__})"

    orchestrator_model, orchestrator_tokenizer = load_orchestrator_with_retries(
        model_name=orchestrator_model_name,
        device=device,
        retries=args.orchestrator_load_retries,
        sleep_s=args.orchestrator_load_retry_sleep,
    )

    decision_cache = {} if args.force_reroute else load_decision_cache(decision_cache_path)

    metrics_avg = MetricAverage()
    rows: list[dict[str, Any]] = []
    fallback_count = 0

    for query_idx, query_text in enumerate(tqdm(unique_queries, desc="Routing+Eval"), start=1):
        indices = query_to_indices.get(query_text, [])
        if not indices:
            continue
        y_true = np.asarray([all_relevant[idx] for idx in indices])

        web_meta = query_to_web_meta.get(query_text, {})
        router_payload = _build_router_input(query=query_text, query_indices=indices)

        if (not args.force_reroute) and query_text in decision_cache:
            decision = decision_cache[query_text]
            chosen_method = str(decision.get("method", fallback_method))
            decision_invalid = bool(decision.get("invalid_decision", False))
            decision_confidence = decision.get("confidence")
            decision_reason = str(decision.get("reason", ""))
            decision_signals = decision.get("signals", [])
            decision_raw = str(decision.get("raw", ""))
        else:
            decision = route_query_method(
                model=orchestrator_model,
                tokenizer=orchestrator_tokenizer,
                router_prompt=router_prompt,
                payload=router_payload,
                max_new_tokens=args.orchestrator_max_new_tokens,
                temperature=args.orchestrator_temperature,
                fallback_method=fallback_method,
            )
            chosen_method = decision["method"]
            decision_invalid = bool(decision.get("invalid_decision", False))
            decision_confidence = decision.get("confidence")
            decision_reason = str(decision.get("reason", ""))
            decision_signals = decision.get("signals", [])
            decision_raw = str(decision.get("raw", ""))
            decision_cache[query_text] = decision

        executed_method = chosen_method
        fallback_used = False
        fallback_reason = ""
        y_pred: np.ndarray | None = None

        # Precompute candidate image embeddings for this query once.
        candidate_ids = [all_image_ids[idx] for idx in indices]
        candidate_embs = torch.stack([image_emb_cache[image_id] for image_id in candidate_ids])

        if chosen_method == "t2i":
            query_key = f"query::{query_text}"
            query_emb = ensure_text_embeddings(
                keys=[query_key],
                key_to_text={query_key: query_text},
                emb_cache=text_emb_cache,
                model=retrieval_model,
                tokenizer=retrieval_tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )[0]
            y_pred = (candidate_embs.float() @ query_emb.float()).numpy()

        elif chosen_method == "i2i":
            web_embs = get_web_embeddings(query_text)
            if web_embs.numel() == 0:
                fallback_used = True
                fallback_reason = "no_valid_web_images_for_query"
            else:
                sim_matrix = candidate_embs.float() @ web_embs.float().T
                y_pred = aggregate_similarity_scores(
                    sim_matrix,
                    mode=args.aggregation,
                    top_k=args.top_k_sims,
                ).numpy()

        elif chosen_method == "t2t":
            try:
                ensure_captions_for_query(query_text)
                query_key = f"query::{query_text}"
                query_emb = ensure_text_embeddings(
                    keys=[query_key],
                    key_to_text={query_key: query_text},
                    emb_cache=text_emb_cache,
                    model=retrieval_model,
                    tokenizer=retrieval_tokenizer,
                    device=device,
                    batch_size=max(1, args.batch_size),
                )[0]

                caption_keys = [_caption_key(image_id, args.caption_vlm_model) for image_id in candidate_ids]
                key_to_text = {key: caption_cache.get(key, "unlabeled image") for key in caption_keys}
                caption_embs = ensure_text_embeddings(
                    keys=caption_keys,
                    key_to_text=key_to_text,
                    emb_cache=text_emb_cache,
                    model=retrieval_model,
                    tokenizer=retrieval_tokenizer,
                    device=device,
                    batch_size=max(1, args.batch_size),
                )
                y_pred = (caption_embs.float() @ query_emb.float()).numpy()
            except Exception as exc:
                fallback_used = True
                fallback_reason = f"t2t_failed:{exc.__class__.__name__}"

        if y_pred is None:
            executed_method = fallback_method
            fallback_used = True
            if executed_method != "t2i":
                raise RuntimeError(
                    f"Fallback is set to '{executed_method}', but only 't2i' fallback is implemented.",
                )
            query_key = f"query::{query_text}"
            query_emb = ensure_text_embeddings(
                keys=[query_key],
                key_to_text={query_key: query_text},
                emb_cache=text_emb_cache,
                model=retrieval_model,
                tokenizer=retrieval_tokenizer,
                device=device,
                batch_size=max(1, args.batch_size),
            )[0]
            y_pred = (candidate_embs.float() @ query_emb.float()).numpy()

        precision, recall, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap * 100, ndcg * 100, mrr])

        if fallback_used:
            fallback_count += 1

        rows.append(
            {
                "query": query_text,
                "model": args.model_key,
                "chosen_method": chosen_method,
                "executed_method": executed_method,
                "fallback_used": int(fallback_used),
                "fallback_method": fallback_method,
                "fallback_reason": fallback_reason,
                "router_invalid_decision": int(decision_invalid),
                "router_confidence": decision_confidence,
                "router_reason": decision_reason,
                "router_signals": json.dumps(decision_signals, ensure_ascii=True),
                "router_raw": decision_raw,
                "num_candidates": len(indices),
                "num_web_images": int(web_meta.get("num_web_images", 0)),
                "num_web_candidates": int(web_meta.get("num_web_candidates", 0)),
                "query_num_tokens": int(router_payload["query_stats"]["num_tokens"]),
                "ap": ap * 100,
                "ndcg": ndcg * 100,
                "mrr": mrr,
                "precision": precision,
                "recall": recall,
            },
        )

        # Save caches progressively for long runs.
        if query_idx % 20 == 0:
            save_decision_cache(decision_cache_path, decision_cache)
            save_caption_cache(caption_cache_path, caption_cache)

    save_decision_cache(decision_cache_path, decision_cache)
    save_caption_cache(caption_cache_path, caption_cache)

    if caption_wrapper is not None:
        if getattr(caption_wrapper, "model", None) is not None:
            del caption_wrapper.model
        if getattr(caption_wrapper, "processor", None) is not None:
            del caption_wrapper.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_df = pd.DataFrame.from_dict(rows)
    save_results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_results_path, index=False)

    if len(results_df) == 0:
        raise RuntimeError("No query rows were produced by orchestrator run.")

    summary_df = (
        results_df.groupby("executed_method", as_index=False)
        .agg(
            ap=("ap", "mean"),
            ndcg=("ndcg", "mean"),
            mrr=("mrr", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            n_queries=("query", "count"),
        )
        .sort_values("ap", ascending=False)
    )
    summary_df.insert(0, "model", args.model_key)
    summary_df["query_share"] = summary_df["n_queries"] / max(1, len(results_df))
    save_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_summary_path, index=False)

    usage_df = (
        results_df.groupby(["chosen_method", "executed_method"], as_index=False)
        .agg(
            n_queries=("query", "count"),
            fallback_rate=("fallback_used", "mean"),
            avg_router_confidence=("router_confidence", "mean"),
            ap=("ap", "mean"),
            ndcg=("ndcg", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values(["chosen_method", "n_queries"], ascending=[True, False])
    )
    usage_df.insert(0, "model", args.model_key)
    usage_df["query_share"] = usage_df["n_queries"] / max(1, len(results_df))
    save_usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_df.to_csv(save_usage_path, index=False)

    chosen_counts = (
        results_df.groupby("chosen_method", as_index=False)
        .agg(n_queries=("query", "count"))
        .rename(columns={"chosen_method": "method"})
    )
    chosen_counts.insert(0, "usage_type", "chosen")
    executed_counts = (
        results_df.groupby("executed_method", as_index=False)
        .agg(n_queries=("query", "count"))
        .rename(columns={"executed_method": "method"})
    )
    executed_counts.insert(0, "usage_type", "executed")
    method_counts_df = pd.concat([chosen_counts, executed_counts], ignore_index=True)
    method_counts_df.insert(0, "model", args.model_key)
    method_counts_df["query_share"] = method_counts_df["n_queries"] / max(1, len(results_df))
    method_counts_df = method_counts_df.sort_values(["usage_type", "n_queries"], ascending=[True, False])
    save_method_counts_path = save_usage_path.with_name(f"{save_usage_path.stem}_method_counts.csv")
    method_counts_df.to_csv(save_method_counts_path, index=False)

    overall_ap = float(results_df["ap"].mean())
    overall_ndcg = float(results_df["ndcg"].mean())
    overall_mrr = float(results_df["mrr"].mean())
    print("\n=== Orchestrator Results ===")
    print(f"Model: {args.model_key}")
    print(f"Queries: {len(results_df)}")
    print(f"Fallbacks: {fallback_count} ({fallback_count / len(results_df):.1%})")
    print(f"AP: {overall_ap:.2f}  nDCG: {overall_ndcg:.2f}  MRR: {overall_mrr:.3f}")
    print("\nBy executed method:")
    print(summary_df.to_string(index=False))
    print("\nBy chosen->executed usage:")
    print(usage_df.to_string(index=False))
    print("\nMethod counts (chosen and executed):")
    print(method_counts_df.to_string(index=False))
    print(f"\nSaved query-level results to: {save_results_path}")
    print(f"Saved method summary to: {save_summary_path}")
    print(f"Saved usage summary to: {save_usage_path}")
    print(f"Saved method counts to: {save_method_counts_path}")
    print(f"Saved decision cache to: {decision_cache_path}")
    print(f"Saved caption cache to: {caption_cache_path}")


if __name__ == "__main__":
    main()
