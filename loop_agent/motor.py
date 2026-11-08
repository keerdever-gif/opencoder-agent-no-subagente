"""Lightweight service result cache and metrics.

Tool selection and request interpretation belong to the
model; this module does not classify text or contain language rules.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    value: Any
    expires_at: float


class ResultCache:
    """LRU + TTL con real prefix invalidation and a memory limit."""

    def __init__(self, max_size: int = 300, ttl: float = 300.0) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl = max(0.0, float(ttl))
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.per_prefix: dict[str, dict[str, int]] = {}

    @staticmethod
    def _key(prefix: str, args: tuple[Any, ...]) -> str:
        raw = "\0".join((str(prefix), *(str(arg) for arg in args)))
        digest = hashlib.blake2b(raw.encode("utf-8", "ignore"), digest_size=16).hexdigest()
        return f"{prefix}:{digest}"

    def _stats_for(self, prefix: str) -> dict[str, int]:
        stats = self.per_prefix.get(prefix)
        if stats is None:
            stats = {"hits": 0, "misses": 0}
            self.per_prefix[prefix] = stats
        return stats

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, entry in self._data.items() if entry.expires_at <= now]
        for key in expired:
            self._data.pop(key, None)

    def get(self, prefix: str, *args: Any) -> Any | None:
        now = time.monotonic()
        key = self._key(prefix, args)
        entry = self._data.get(key)
        stats = self._stats_for(prefix)
        if entry is None or entry.expires_at <= now:
            if entry is not None:
                self._data.pop(key, None)
            self.misses += 1
            stats["misses"] += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        stats["hits"] += 1
        return entry.value

    def set(self, prefix: str, value: Any, *args: Any) -> None:
        now = time.monotonic()
        self._purge_expired(now)
        key = self._key(prefix, args)
        self._data[key] = CacheEntry(value=value, expires_at=now + self.ttl)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def invalidate(self, prefix: str | None = None) -> int:
        if prefix is None:
            count = len(self._data)
            self._data.clear()
            return count
        marker = f"{prefix}:"
        keys = [key for key in self._data if key.startswith(marker)]
        for key in keys:
            self._data.pop(key, None)
        return len(keys)

    def stats(self) -> dict[str, Any]:
        self._purge_expired(time.monotonic())
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._data),
            "max_size": self.max_size,
            "per_prefix": {key: dict(value) for key, value in self.per_prefix.items()},
        }


class SlidingWindow:
    __slots__ = ("_samples",)

    def __init__(self, size: int = 100) -> None:
        self._samples: deque[float] = deque(maxlen=max(1, size))

    def add(self, value: float) -> None:
        self._samples.append(float(value))

    def percentile(self, percentile: float) -> float:
        if not self._samples:
            return 0.0
        values = sorted(self._samples)
        index = max(0, min(len(values) - 1, int(round(percentile * (len(values) - 1)))))
        return values[index]

    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0


@dataclass(slots=True)
class SensorStats:
    calls: int = 0
    errors: int = 0
    total_latency: float = 0.0
    window: SlidingWindow = field(default_factory=SlidingWindow)
    last_error: str = ""
    last_error_at: float = 0.0
    last_ok_at: float = 0.0
    consecutive_errors: int = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.calls if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        return 1.0 - self.errors / self.calls if self.calls else 1.0

    def observe(self, latency: float, error: bool = False) -> None:
        self.calls += 1
        self.total_latency += max(0.0, latency)
        self.window.add(latency)
        if error:
            self.errors += 1
            self.consecutive_errors += 1
            self.last_error_at = time.time()
        else:
            self.consecutive_errors = 0
            self.last_ok_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "ok_rate": round(self.success_rate, 3),
            "avg_ms": round(self.avg_latency * 1000, 1),
            "p50_ms": round(self.window.percentile(0.50) * 1000, 1),
            "p95_ms": round(self.window.percentile(0.95) * 1000, 1),
            "streak": self.consecutive_errors,
            "last_error": self.last_error,
        }


class Metrics:
    """Bounded in-memory metrics; events and samples use finite windows."""

    def __init__(self) -> None:
        self.per_model: dict[str, SensorStats] = {}
        self.per_key: dict[int, SensorStats] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.t_start = time.time()

    @staticmethod
    def _get(bucket: dict, key: Any) -> SensorStats:
        stats = bucket.get(key)
        if stats is None:
            stats = SensorStats()
            bucket[key] = stats
        return stats

    def record(
        self,
        *,
        model: str | None = None,
        key_index: int | None = None,
        latency: float,
        error: bool = False,
        error_msg: str = "",
    ) -> None:
        if model:
            stats = self._get(self.per_model, model)
            stats.observe(latency, error)
            if error and error_msg:
                stats.last_error = str(error_msg)[:200]
        if key_index is not None:
            self._get(self.per_key, key_index).observe(latency, error)
        self.events.append({
            "t": round(time.time(), 3),
            "model": model,
            "latency_ms": round(max(0.0, latency) * 1000, 1),
            "error": bool(error),
        })

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self.t_start, 1),
            "models": {key: stats.to_dict() for key, stats in self.per_model.items()},
            "keys": {str(key): stats.to_dict() for key, stats in self.per_key.items()},
            "events": list(self.events)[-20:],
        }


class Motor:
    """Stable interface for consumers: cache + metrics."""

    def __init__(self) -> None:
        self.metrics = Metrics()
        self.cache = ResultCache(max_size=300, ttl=300.0)

    def cache_get_search(self, query: str) -> Any | None:
        return self.cache.get("search", query.lower().strip())

    def cache_set_search(self, query: str, value: Any) -> None:
        self.cache.set("search", value, query.lower().strip())

    def cache_get_fetch(self, url: str) -> Any | None:
        return self.cache.get("fetch", url.strip())

    def cache_set_fetch(self, url: str, value: Any) -> None:
        self.cache.set("fetch", value, url.strip())

    def record(
        self,
        *,
        model: str | None = None,
        key_index: int | None = None,
        latency: float,
        error: bool = False,
        error_msg: str = "",
    ) -> None:
        self.metrics.record(
            model=model,
            key_index=key_index,
            latency=latency,
            error=error,
            error_msg=error_msg,
        )

    def snapshot(self) -> dict[str, Any]:
        return {"cache": self.cache.stats(), **self.metrics.snapshot()}


motor = Motor()
