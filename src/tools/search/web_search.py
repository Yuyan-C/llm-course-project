from __future__ import annotations

"""
Web search tool helpers used by the repository's tool-calling pipeline.

This module intentionally provides a stable, JSON-serializable output schema so it can
be safely embedded into tool responses passed back to an LLM conversation.

Search strategy:
1. Preferred provider: `ddgs` (or legacy `duckduckgo_search`) for regular web search.
2. Fallback provider: DuckDuckGo Instant Answer API when DDGS is unavailable.

Why two providers:
- DDGS generally returns richer web results (title/link/snippet).
- Some runtime environments may not have DDGS installed or may block one endpoint.
- The fallback path still gives a best-effort answer instead of hard-failing.

Returned object shape:
{
    "query": "<original query>",
    "provider": "<ddgs|duckduckgo_search|duckduckgo_instant_answer|none>",
    "results": [
        {"position": 1, "title": "...", "url": "...", "snippet": "...", "source": "..."}
    ],
    # optional:
    "warnings": [...],
    "error": "...",
    "details": {...}
}
"""

from typing import Any, Dict, Iterable, List
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError


def _normalize_search_item(item: Dict[str, Any], position: int) -> Dict[str, Any]:
    """
    Normalize a result item from different providers into one common schema.

    Different search backends use different field names (`href` vs `url`, `body`
    vs `snippet`, etc.). This helper maps the common alternatives into one stable
    dictionary that downstream code can rely on.

    Args:
        item: Raw result dictionary from a backend response.
        position: 1-based rank used in returned results.

    Returns:
        Normalized dictionary with keys: `position`, `title`, `url`, `snippet`,
        and `source`.
    """
    title = item.get("title") or item.get("heading") or ""
    url = item.get("href") or item.get("url") or item.get("link") or ""
    snippet = item.get("body") or item.get("snippet") or item.get("description") or item.get("text") or ""
    source = item.get("source")

    return {
        "position": position,
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
    }


def _iter_related_topics(topics: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    Flatten nested `RelatedTopics` entries from the Instant Answer payload.

    DuckDuckGo Instant Answer may return a tree where some entries contain a
    `Topics` list. This generator recursively yields only leaf topic entries.

    Args:
        topics: Top-level `RelatedTopics` list from DuckDuckGo payload.

    Yields:
        Leaf topic dictionaries (each expected to contain `Text` and `FirstURL`).
    """
    for topic in topics:
        if "Topics" in topic:
            yield from _iter_related_topics(topic["Topics"])
        else:
            yield topic


def _search_with_ddgs(
    query: str,
    max_results: int,
    region: str,
    safesearch: str,
    backend: str,
) -> Dict[str, Any]:
    """
    Execute web search via DDGS-compatible client.

    Import resolution order:
    1. `ddgs` (new package name)
    2. `duckduckgo_search` (older package name still used in some setups)

    The DDGS API has changed across versions; some versions accept `backend`,
    others do not. This function handles both signatures.

    Args:
        query: User query.
        max_results: Maximum number of records to keep.
        region: Regional hint (e.g., `us-en`).
        safesearch: Safe-search mode.
        backend: DDGS backend preference when supported.

    Returns:
        Dictionary containing `provider` and normalized `results`.

    Raises:
        ImportError: If neither DDGS package is available.
        Exception: Any network/provider/client failure from DDGS internals.
    """
    ddgs_cls = None
    provider = ""

    try:
        from ddgs import DDGS

        ddgs_cls = DDGS
        provider = "ddgs"
    except Exception:
        try:
            from duckduckgo_search import DDGS

            ddgs_cls = DDGS
            provider = "duckduckgo_search"
        except Exception as exc:
            raise ImportError(
                "Could not import ddgs or duckduckgo_search. Install with `uv pip install ddgs`."
            ) from exc

    client = ddgs_cls()
    try:
        # Newer clients often support the `backend` parameter. If not, retry
        # without it for compatibility with older package versions.
        try:
            raw_results = client.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
                backend=backend,
            )
        except TypeError:
            raw_results = client.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )

        # Some versions return a list, others may return an iterator/generator.
        # Materialize at most `max_results` to keep output bounded.
        if isinstance(raw_results, list):
            records = raw_results[:max_results]
        else:
            records = []
            for item in raw_results:
                records.append(item)
                if len(records) >= max_results:
                    break
    finally:
        # Not all client versions expose `close()`. If available, call it to
        # release HTTP resources.
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "provider": provider,
        "results": [_normalize_search_item(item, idx) for idx, item in enumerate(records, start=1)],
    }


def _search_with_instant_answer(query: str, max_results: int) -> Dict[str, Any]:
    """
    Run a fallback search using DuckDuckGo's public Instant Answer endpoint.

    This endpoint is lighter than full search APIs and may return fewer links,
    but it is useful as a no-key fallback when DDGS is unavailable.

    Args:
        query: User query.
        max_results: Maximum number of items to return.

    Returns:
        Dictionary with `provider` and normalized `results`.
    """
    # Request JSON, skip HTML/redirection/disambiguation noise for cleaner output.
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
        "skip_disambig": 1,
    }
    request = Request(
        f"https://api.duckduckgo.com/?{urlencode(params)}",
        headers={"User-Agent": "llm-course-project/1.0"},
    )
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results: List[Dict[str, Any]] = []

    # If the API has a main abstract answer, surface it as top-ranked result.
    abstract_text = payload.get("AbstractText", "")
    abstract_url = payload.get("AbstractURL", "")
    heading = payload.get("Heading", "")
    if abstract_text and abstract_url:
        results.append(
            {
                "position": 1,
                "title": heading or query,
                "url": abstract_url,
                "snippet": abstract_text,
                "source": "DuckDuckGo Instant Answer",
            }
        )

    # Then append related topic links, flattened from nested groups.
    for topic in _iter_related_topics(payload.get("RelatedTopics", [])):
        text = topic.get("Text", "")
        first_url = topic.get("FirstURL", "")
        if not text or not first_url:
            continue

        # Many entries look like "Title - description"; keep a concise title.
        title = text.split(" - ", 1)[0].strip()
        results.append(
            {
                "position": len(results) + 1,
                "title": title,
                "url": first_url,
                "snippet": text,
                "source": "DuckDuckGo Instant Answer",
            }
        )
        if len(results) >= max_results:
            break

    return {
        "provider": "duckduckgo_instant_answer",
        "results": results[:max_results],
    }


def run_web_search(
    query: str,
    max_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    backend: str = "duckduckgo",
) -> Dict[str, Any]:
    """
    Search the web and return structured top results for tool-calling.

    This is the public entrypoint intended to be exposed to the LLM as a tool.
    It always returns a dictionary so callers can serialize/log the response
    even when providers fail.

    Resolution order:
    1. Try DDGS (`ddgs` or `duckduckgo_search`) for richer web search.
    2. Fall back to DuckDuckGo Instant Answer endpoint.
    3. If both fail, return an error payload with per-provider details.

    Args:
        query: Search query string. Must be non-empty after stripping whitespace.
        max_results: Max number of results to return, clamped to [1, 20].
        region: Region hint passed to DDGS (ignored by fallback provider).
        safesearch: Safe-search setting passed to DDGS.
        backend: DDGS backend hint (version-dependent support).

    Returns:
        A dictionary containing:
        - `query`: normalized input query
        - `provider`: provider used (`ddgs`, `duckduckgo_search`,
          `duckduckgo_instant_answer`, or `none`)
        - `results`: normalized result list
        Optional keys:
        - `warnings`: fallback warning(s)
        - `error`: top-level failure message
        - `details`: provider-level exception strings

    Raises:
        ValueError: If `query` is empty after stripping.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must be a non-empty string.")

    # Keep bounded output size to avoid overwhelming downstream LLM context.
    max_results = max(1, min(max_results, 20))
    errors: Dict[str, str] = {}

    try:
        ddgs_response = _search_with_ddgs(
            query=cleaned_query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            backend=backend,
        )
        return {"query": cleaned_query, **ddgs_response}
    except Exception as exc:
        # Preserve error text for observability while still attempting fallback.
        errors["ddgs"] = str(exc)

    try:
        fallback_response = _search_with_instant_answer(cleaned_query, max_results=max_results)
        return {
            "query": cleaned_query,
            **fallback_response,
            "warnings": ["DDGS search unavailable. Used DuckDuckGo Instant Answer fallback."],
        }
    except Exception as exc:
        # Final failure path: return structured error payload instead of raising.
        errors["fallback"] = str(exc)
        return {
            "query": cleaned_query,
            "provider": "none",
            "results": [],
            "error": "All web search providers failed.",
            "details": errors,
        }


def _normalize_image_item(item: Dict[str, Any], position: int) -> Dict[str, Any]:
    title = item.get("title") or item.get("heading") or ""
    url = item.get("url") or item.get("link") or item.get("image") or ""
    image = item.get("image") or item.get("url") or ""
    thumbnail = item.get("thumbnail") or item.get("thumb") or ""
    source = item.get("source") or item.get("provider")
    width = item.get("width")
    height = item.get("height")

    return {
        "position": position,
        "title": title,
        "url": url,
        "image": image,
        "thumbnail": thumbnail,
        "source": source,
        "width": width,
        "height": height,
    }


def _shorten_query(query: str, max_words: int = 6) -> str:
    words = query.split()
    if len(words) <= max_words:
        return query
    return " ".join(words[:max_words])


def run_web_image_search(
    query: str,
    max_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    backend: str = "duckduckgo",
) -> Dict[str, Any]:
    """
    Search the web for images using DDGS and return structured results.

    Args:
        query: Search query string. Must be non-empty after stripping whitespace.
        max_results: Max number of results to return, clamped to [1, 20].
        region: Region hint passed to DDGS.
        safesearch: Safe-search setting passed to DDGS.
        backend: DDGS backend hint (version-dependent support).

    Returns:
        {
            "query": "...",
            "provider": "ddgs"|"duckduckgo_search"|"none",
            "results": [ {"image": "...", ...}, ...],
            "error": "..." (optional)
        }
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must be a non-empty string.")

    max_results = max(1, min(max_results, 20))
    errors: Dict[str, str] = {}

    try:
        ddgs_cls = None
        provider = ""
        try:
            from ddgs import DDGS

            ddgs_cls = DDGS
            provider = "ddgs"
        except Exception:
            from duckduckgo_search import DDGS

            ddgs_cls = DDGS
            provider = "duckduckgo_search"

        def _fetch_images(client: Any, query_text: str) -> List[Dict[str, Any]]:
            try:
                raw_results = client.images(
                    query_text,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results,
                    backend=backend,
                )
            except TypeError:
                raw_results = client.images(
                    query_text,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results,
                )

            if isinstance(raw_results, list):
                return raw_results[:max_results]

            records: List[Dict[str, Any]] = []
            for item in raw_results:
                records.append(item)
                if len(records) >= max_results:
                    break
            return records

        client = ddgs_cls()
        try:
            records = _fetch_images(client, cleaned_query)
            warnings: List[str] = []
            shortened_query = _shorten_query(cleaned_query)
            if not records and shortened_query != cleaned_query:
                records = _fetch_images(client, shortened_query)
                if records:
                    warnings.append(
                        "No results for the original query. Retried with a shorter query."
                    )
            response = {
                "query": cleaned_query,
                "provider": provider,
                "results": [_normalize_image_item(item, idx) for idx, item in enumerate(records, start=1)],
            }
            if warnings:
                response["warnings"] = warnings
            return response
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
    except Exception as exc:
        if "No results found" in str(exc):
            shortened_query = _shorten_query(cleaned_query)
            if shortened_query != cleaned_query:
                try:
                    client = ddgs_cls()
                    try:
                        records = _fetch_images(client, shortened_query)
                    finally:
                        close_fn = getattr(client, "close", None)
                        if callable(close_fn):
                            close_fn()
                    return {
                        "query": cleaned_query,
                        "provider": provider or "ddgs",
                        "results": [_normalize_image_item(item, idx) for idx, item in enumerate(records, start=1)],
                        "warnings": [
                            "No results for the original query. Retried with a shorter query."
                        ],
                    }
                except Exception:
                    pass
            return {
                "query": cleaned_query,
                "provider": provider or "ddgs",
                "results": [],
                "warnings": ["No image results found for query."],
            }
        errors["ddgs"] = str(exc)

    return {
        "query": cleaned_query,
        "provider": "none",
        "results": [],
        "error": "All image search providers failed.",
        "details": errors,
    }
