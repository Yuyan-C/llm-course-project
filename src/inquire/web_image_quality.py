from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from PIL import Image
import torch


STOCK_PHOTO_DOMAINS = {
    "alamy.com",
    "shutterstock.com",
    "gettyimages.com",
    "istockphoto.com",
    "dreamstime.com",
    "depositphotos.com",
    "adobestock.com",
}

BLOCKED_METADATA_KEYWORDS = (
    "shutterstock",
    "gettyimages",
    "istockphoto",
    "dreamstime",
    "depositphotos",
    "adobestock",
    "watermark",
    "royalty free",
    "stock photo",
    "vector illustration",
    "clipart",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@dataclass
class WebImageQualityConfig:
    min_short_side: int = 224
    max_aspect_ratio: float = 2.8
    stock_thumbnail_short_side: int = 600
    metadata_relevance_threshold: float = 0.25
    drop_stock_thumbnails: bool = True


def _normalize_domain(value: str | None) -> str:
    if not value:
        return ""
    domain = value.strip().lower()
    if "://" in domain:
        domain = urlparse(domain).netloc.lower()
    domain = domain.split("/", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _extract_search_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    tool_result = record.get("tool_result")
    if not isinstance(tool_result, dict):
        return []
    raw_results = tool_result.get("results")
    if not isinstance(raw_results, list):
        return []
    return [item for item in raw_results if isinstance(item, dict)]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _metadata_relevance_score(query: str, metadata_text: str) -> float:
    query_tokens = [tok for tok in _tokenize(query) if len(tok) >= 3 and tok not in STOPWORDS]
    if not query_tokens:
        return 0.0

    metadata_tokens = set(_tokenize(metadata_text))
    if not metadata_tokens:
        return 0.0

    overlap = sum(1 for tok in query_tokens if tok in metadata_tokens)
    token_match_score = overlap / len(query_tokens)

    normalized_query = " ".join(_tokenize(query))
    normalized_metadata = " ".join(_tokenize(metadata_text))
    phrase_boost = 0.2 if normalized_query and normalized_query in normalized_metadata else 0.0

    return min(1.0, token_match_score + phrase_boost)


def _first_blocked_keyword(metadata_text: str) -> str | None:
    lowered = metadata_text.lower()
    for keyword in BLOCKED_METADATA_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


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


def extract_image_embedding_tensor(model_output: Any) -> torch.Tensor:
    return _extract_embedding_tensor(model_output, "image_embeds")


def inspect_downloaded_image(
    path: Path,
    config: WebImageQualityConfig,
    *,
    query: str | None = None,
    url: str | None = None,
    source: str | None = None,
    title: str | None = None,
    thumbnail: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {
        "path": str(path),
        "query": query,
        "title": title,
        "url": url,
        "source": source,
        "thumbnail": thumbnail,
    }

    if not path.exists():
        info["reject_reason"] = "missing_file"
        return False, info

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
    except Exception as exc:
        info["reject_reason"] = f"corrupted_or_unreadable:{exc.__class__.__name__}"
        return False, info

    if width <= 0 or height <= 0:
        info["reject_reason"] = "invalid_dimensions"
        return False, info

    short_side = min(width, height)
    long_side = max(width, height)
    aspect_ratio = float(long_side / short_side)

    info["width"] = width
    info["height"] = height
    info["short_side"] = short_side
    info["aspect_ratio"] = aspect_ratio

    if short_side < config.min_short_side:
        info["reject_reason"] = f"tiny_image_short_side<{config.min_short_side}"
        return False, info

    if aspect_ratio > config.max_aspect_ratio:
        info["reject_reason"] = f"aspect_ratio>{config.max_aspect_ratio}"
        return False, info

    domain = _normalize_domain(source) or _normalize_domain(url)
    info["domain"] = domain

    if (
        config.drop_stock_thumbnails
        and domain in STOCK_PHOTO_DOMAINS
        and short_side < config.stock_thumbnail_short_side
    ):
        info["reject_reason"] = f"likely_stock_thumbnail_short_side<{config.stock_thumbnail_short_side}"
        return False, info

    metadata_text = " ".join(
        value
        for value in (
            title or "",
            url or "",
            source or "",
            thumbnail or "",
            domain,
        )
        if value
    )

    keyword = _first_blocked_keyword(metadata_text)
    if keyword is not None:
        info["blocked_metadata_keyword"] = keyword
        info["reject_reason"] = f"blocked_metadata_keyword:{keyword}"
        return False, info

    metadata_relevance = _metadata_relevance_score(query or "", metadata_text)
    info["metadata_relevance"] = metadata_relevance
    info["quality_score"] = float(short_side + 300.0 * metadata_relevance)

    if query and metadata_relevance < config.metadata_relevance_threshold:
        info["reject_reason"] = (
            "low_metadata_relevance<"
            f"{config.metadata_relevance_threshold}"
        )
        return False, info

    return True, info


def filter_record_images(
    record: dict[str, Any],
    config: WebImageQualityConfig,
    max_keep: int | None = None,
    min_keep: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    query = str(record.get("query", ""))
    image_paths = [p for p in record.get("image_paths", []) if isinstance(p, str)]
    image_urls = [u for u in record.get("image_urls", []) if isinstance(u, str)]

    search_items = _extract_search_items(record)
    search_item_by_url: dict[str, dict[str, Any]] = {}
    for item in search_items:
        item_url = item.get("image") or item.get("url")
        if isinstance(item_url, str) and item_url:
            search_item_by_url[item_url] = item

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, path_str in enumerate(image_paths):
        url = image_urls[idx] if idx < len(image_urls) else None

        item = search_items[idx] if idx < len(search_items) else None
        if not isinstance(item, dict) and isinstance(url, str):
            item = search_item_by_url.get(url)
        if not isinstance(item, dict):
            item = {}

        if not url:
            item_url = item.get("image") or item.get("url")
            url = item_url if isinstance(item_url, str) else None

        title = str(item.get("title") or item.get("heading") or "")
        source = str(item.get("source") or item.get("provider") or "")
        thumbnail = str(item.get("thumbnail") or item.get("thumb") or "")

        keep, info = inspect_downloaded_image(
            Path(path_str),
            config,
            query=query,
            url=url,
            source=source,
            title=title,
            thumbnail=thumbnail,
        )
        if keep:
            accepted.append(info)
        else:
            rejected.append(info)

    accepted = sorted(
        accepted,
        key=lambda item: float(item.get("quality_score", 0.0)),
        reverse=True,
    )

    if max_keep is not None and max_keep >= 0 and len(accepted) > max_keep:
        overflow = accepted[max_keep:]
        accepted = accepted[:max_keep]
        for item in overflow:
            item_copy = dict(item)
            item_copy["reject_reason"] = "exceeded_keep_k_after_quality_ranking"
            rejected.append(item_copy)

    target_min = 0 if min_keep is None else max(0, min_keep)
    if max_keep is not None and max_keep >= 0:
        target_min = min(target_min, max_keep)

    if len(accepted) < target_min:
        rescue_candidates = []
        for item in rejected:
            reason = str(item.get("reject_reason", ""))
            if not reason.startswith("low_metadata_relevance<"):
                continue
            rescue_candidates.append(item)

        rescue_candidates = sorted(
            rescue_candidates,
            key=lambda item: float(item.get("quality_score", 0.0)),
            reverse=True,
        )
        to_rescue = rescue_candidates[: target_min - len(accepted)]
        rescue_keys = {(item.get("path"), item.get("reject_reason")) for item in to_rescue}

        rescued_items: list[dict[str, Any]] = []
        for item in to_rescue:
            rescued = dict(item)
            rescued["rescued_from_reject_reason"] = rescued.get("reject_reason")
            rescued.pop("reject_reason", None)
            rescued_items.append(rescued)
        accepted.extend(rescued_items)

        kept_rejected = []
        for item in rejected:
            key = (item.get("path"), item.get("reject_reason"))
            if key in rescue_keys:
                continue
            kept_rejected.append(item)
        rejected = kept_rejected

        accepted = sorted(
            accepted,
            key=lambda item: float(item.get("quality_score", 0.0)),
            reverse=True,
        )

    accepted_paths = [item["path"] for item in accepted]
    return accepted_paths, accepted, rejected
