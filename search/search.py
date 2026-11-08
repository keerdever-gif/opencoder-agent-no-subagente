"""DuckDuckGo search with caching, retries, and URL validation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_QUERY_CHARS = 500
MAX_RESULTS = 20
DEFAULT_REGION = "wt-wt"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_CACHE_TTL_S = 120.0
DEFAULT_CACHE_SIZE = 256
DEFAULT_RETRIES = 2
DEFAULT_MAX_CONCURRENCY = 5

_REGION_RE = re.compile(r"^(wt-wt|[a-z]{2}-[a-z]{2})$")
_LOG = logging.getLogger("search")

# The semaphore limits concurrent queries within the process.
_SEMAPHORE = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)

# Thread-safe LRU cache: the provider runs through asyncio.to_thread.
_CACHE: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


class SearchError(Exception):
    """Controlled search engine error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass
class SearchResult:
    """Normalized search result. The first four fields remain backward-compatible."""

    rank: int
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "duckduckgo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_rag_document(self) -> dict[str, Any]:
        """Convert the result to a document for RAG pipelines."""
        return {
            "id": hashlib.sha256(self.url.encode("utf-8")).hexdigest(),
            "text": "\n".join(part for part in (self.title, self.snippet) if part),
            "metadata": {
                "title": self.title,
                "url": self.url,
                "rank": self.rank,
                "score": self.score,
                "source": self.source,
            },
        }


@dataclass
class SearchConfig:
    """Optional configuration; public functions also accept direct arguments."""

    timeout_s: float = DEFAULT_TIMEOUT_S
    retries: int = DEFAULT_RETRIES
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S
    cache_size: int = DEFAULT_CACHE_SIZE
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY


def _validate(
    query: object,
    max_results: object,
    region: str | None,
    safesearch: str | None,
    timeout_s: float,
    retries: int,
) -> tuple[str, int, str, str]:
    if not isinstance(query, str) or not query.strip():
        raise SearchError("invalid_query", "The search query cannot be empty.")
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        raise SearchError(
            "invalid_query", f"The search query exceeds {MAX_QUERY_CHARS} caracteres."
        )
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise SearchError("invalid_max_results", "max_results must be an integer >= 1.")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise SearchError("invalid_timeout", "timeout_s must be a positive finite number.")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0 or retries > 5:
        raise SearchError("invalid_retries", "retries must be an integer between 0 and 5.")

    region = (region or DEFAULT_REGION).lower()
    if not _REGION_RE.fullmatch(region):
        raise SearchError("invalid_region", "Region must use a value such as 'us-en', 'es-es' o 'wt-wt'.")
    safesearch = (safesearch or DEFAULT_SAFESEARCH).lower()
    if safesearch not in {"on", "moderate", "off"}:
        raise SearchError("invalid_safesearch", "safesearch must be: on, moderate, or off.")

    return query, min(max_results, MAX_RESULTS), region, safesearch


def _cache_key(query: str, max_results: int, region: str, safesearch: str) -> str:
    raw = f"{query.casefold()}|{max_results}|{region}|{safesearch}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str, ttl_s: float) -> list[dict[str, Any]] | None:
    if ttl_s <= 0:
        return None
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        created, value = entry
        if now - created > ttl_s:
            del _CACHE[key]
            return None
        _CACHE.move_to_end(key)
        return [dict(item) for item in value]


def _cache_put(key: str, value: list[dict[str, Any]], ttl_s: float, max_size: int) -> None:
    if ttl_s <= 0 or max_size <= 0:
        return
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), [dict(item) for item in value])
        _CACHE.move_to_end(key)
        while len(_CACHE) > max_size:
            _CACHE.popitem(last=False)


def clear_cache() -> None:
    """Clears the in-memory cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _canonical_url(url: object) -> str | None:
    """Accepts only HTTP(S), and removes fragments and embedded credentials."""
    if not isinstance(url, str) or len(url) > 4096:
        return None
    try:
        parsed = urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        # Rejects URLs without a valid host or with invalid ports.
        _ = parsed.port
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        netloc = host
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
    except (ValueError, UnicodeError):
        return None


def _ddgs_text(
    query: str,
    max_results: int,
    region: str,
    safesearch: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise SearchError(
            "search_dependency_missing",
            "The 'ddgs' package is missing. Install it with: pip install ddgs",
        ) from exc

    rows = DDGS(timeout=max(1, int(timeout_s))).text(
        query,
        region=region,
        safesearch=safesearch,
        max_results=max_results,
    )
    return list(rows or [])


def _normalise(rows: list[dict[str, Any]], max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()

    for item in rows or []:
        if not isinstance(item, dict):
            continue
        url = _canonical_url(item.get("href") or item.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)

        title = str(item.get("title") or "").strip()[:500]
        snippet = str(item.get("body") or item.get("snippet") or "").strip()[:2000]
        if not title and not snippet:
            continue

        results.append(
            SearchResult(
                rank=len(results) + 1,
                title=title,
                url=url,
                snippet=snippet,
                score=0.0,
            )
        )
        if len(results) >= max_results:
            break

    return results


def _query_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z0-9_-]+", text, flags=re.UNICODE)
        if len(token) > 2
    }


def _score_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Transparent heuristic scoring; not a replacement for an evaluated reranker."""
    query_terms = _query_terms(query)
    if not query_terms:
        return results

    for result in results:
        title_terms = _query_terms(result.title)
        body_terms = _query_terms(result.snippet)
        title_match = len(query_terms & title_terms) / len(query_terms)
        body_match = len(query_terms & body_terms) / len(query_terms)
        result.score = round(min(1.0, 0.65 * title_match + 0.35 * body_match), 4)

    # Stable ordering: score, then original position.
    results.sort(key=lambda item: (-item.score, item.rank))
    for index, result in enumerate(results, start=1):
        result.rank = index
    return results


async def search(
    query: str,
    max_results: int = 5,
    region: str | None = None,
    safesearch: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    retries: int = DEFAULT_RETRIES,
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    cache_size: int = DEFAULT_CACHE_SIZE,
    use_cache: bool = True,
    rerank: bool = True,
) -> list[SearchResult]:
    """Searches DuckDuckGo asynchronously and returns normalized results."""
    query, max_results, region, safesearch = _validate(
        query, max_results, region, safesearch, timeout_s, retries
    )
    key = _cache_key(query, max_results, region, safesearch)

    if use_cache:
        cached = _cache_get(key, cache_ttl_s)
        if cached is not None:
            results = _normalise(cached, max_results)
            return _score_results(query, results) if rerank else results

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with _SEMAPHORE:
                rows = await asyncio.wait_for(
                    asyncio.to_thread(
                        _ddgs_text, query, max_results, region, safesearch, timeout_s
                    ),
                    timeout=timeout_s + 2,
                )
            if use_cache:
                _cache_put(key, rows, cache_ttl_s, cache_size)
            results = _normalise(rows, max_results)
            return _score_results(query, results) if rerank else results
        except asyncio.TimeoutError as exc:
            last_error = exc
            code = "search_timeout"
        except SearchError:
            raise
        except Exception as exc:
            last_error = exc
            code = "search_provider_unavailable"

        if attempt < retries:
            delay = min(4.0, 0.4 * (2**attempt)) + random.uniform(0, 0.15)
            _LOG.warning("Search failed (%s); retry %s/%s", code, attempt + 1, retries)
            await asyncio.sleep(delay)

    if code == "search_timeout":
        raise SearchError("search_timeout", "DuckDuckGo did not respond before the timeout.") from last_error
    raise SearchError(
        "search_provider_unavailable",
        "Search failed or DuckDuckGo temporarily rate-limited requests.",
    ) from last_error


def search_sync(
    query: str,
    max_results: int = 5,
    region: str | None = None,
    safesearch: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    **kwargs: Any,
) -> list[SearchResult]:
    """Synchronous version. Do not call from an already-running asyncio loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            search(query, max_results, region, safesearch, timeout_s, **kwargs)
        )
    raise SearchError(
        "event_loop_running",
        "An asyncio loop is already running. Use: results = await search(...).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search engine for AI agents")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--max-results", type=int, default=5)
    parser.add_argument("--region", default=None)
    parser.add_argument("--safesearch", default=None, choices=["on", "moderate", "off"])
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rag", action="store_true", help="Devuelve documentos preparados para RAG")
    parser.add_argument("--pretty", action="store_true", help="JSON indentado")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        results = search_sync(
            args.query,
            max_results=args.max_results,
            region=args.region,
            safesearch=args.safesearch,
            timeout_s=args.timeout,
            retries=args.retries,
            use_cache=not args.no_cache,
            rerank=not args.no_rerank,
        )
        payload: dict[str, Any] = {
            "query": args.query,
            "result_count": len(results),
            "results": (
                [result.to_rag_document() for result in results]
                if args.rag
                else [result.to_dict() for result in results]
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0
    except SearchError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
