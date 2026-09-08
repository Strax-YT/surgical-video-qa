"""Dynamic model loading with an LRU-bounded resident set.

Why this exists: YOLOv8n is ~6 MB but SAM ViT-H is ~2.4 GB. Loading every
variant at startup either blows the container memory limit or makes cold start
unusable, and loading on every request wastes seconds of GPU transfer. So
models are registered as *loader callables*, materialised on first use, and
evicted least-recently-used once the resident set hits capacity.

Two details that matter under a real request load:

1.  Loading happens under a *per-key* lock, not a global one. Two concurrent
    requests for the same model wait for one load; a request for a different
    model is not blocked behind it.
2.  The cache is only mutated after a successful load, so a loader that raises
    (missing weights, OOM) leaves no poisoned entry behind.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ModelNotRegisteredError(KeyError):
    """Asked for a model key that was never registered."""


@dataclass
class ModelHandle[T]:
    """A loaded model plus the bookkeeping the /models endpoint reports on."""

    key: str
    model: T
    load_time_ms: float
    loaded_at: float
    hits: int = 0
    last_used_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.hits += 1
        self.last_used_at = time.time()


class ModelRegistry:
    """LRU-bounded registry of lazily-loaded models."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._resident: OrderedDict[str, ModelHandle[Any]] = OrderedDict()
        self._registry_lock = threading.RLock()   # guards _loaders / _resident
        self._key_locks: dict[str, threading.Lock] = {}
        self._evictions = 0
        self._misses = 0

    # ---------------------------------------------------------------- register

    def register(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        """Declare a model without loading it."""
        with self._registry_lock:
            if key in self._loaders and not replace:
                raise ValueError(f"model {key!r} already registered")
            self._loaders[key] = loader
            self._metadata[key] = metadata or {}
            self._key_locks.setdefault(key, threading.Lock())
            # A replaced loader invalidates anything already resident.
            if replace:
                self._resident.pop(key, None)
        logger.debug("registered model %s", key)

    def registered_keys(self) -> list[str]:
        with self._registry_lock:
            return sorted(self._loaders)

    def resident_keys(self) -> list[str]:
        """Most-recently-used last, matching eviction order."""
        with self._registry_lock:
            return list(self._resident)

    def is_resident(self, key: str) -> bool:
        with self._registry_lock:
            return key in self._resident

    # -------------------------------------------------------------------- get

    def get(self, key: str) -> Any:
        """Return the model for `key`, loading it if necessary."""
        return self.handle(key).model

    def handle(self, key: str) -> ModelHandle[Any]:
        with self._registry_lock:
            if key not in self._loaders:
                raise ModelNotRegisteredError(
                    f"{key!r} not registered; available: {sorted(self._loaders)}"
                )
            handle = self._resident.get(key)
            if handle is not None:
                self._resident.move_to_end(key)
                handle.touch()
                return handle
            key_lock = self._key_locks[key]
            loader = self._loaders[key]
            metadata = dict(self._metadata[key])

        # Load outside the registry lock so other keys stay servable, but under
        # the per-key lock so we never load the same weights twice.
        with key_lock:
            with self._registry_lock:
                handle = self._resident.get(key)
                if handle is not None:  # another thread won the race
                    self._resident.move_to_end(key)
                    handle.touch()
                    return handle

            logger.info("cold-loading model %s", key)
            started = time.perf_counter()
            model = loader()  # may raise; nothing is cached if it does
            load_time_ms = (time.perf_counter() - started) * 1000.0
            logger.info("loaded model %s in %.0f ms", key, load_time_ms)

            handle = ModelHandle(
                key=key,
                model=model,
                load_time_ms=load_time_ms,
                loaded_at=time.time(),
                metadata=metadata,
            )
            handle.touch()

            with self._registry_lock:
                self._resident[key] = handle
                self._resident.move_to_end(key)
                self._misses += 1
                self._evict_if_needed()
            return handle

    # ----------------------------------------------------------------- unload

    def unload(self, key: str) -> bool:
        """Drop one model from memory. Returns False if it wasn't resident."""
        with self._registry_lock:
            handle = self._resident.pop(key, None)
        if handle is None:
            return False
        self._release(handle)
        return True

    def clear(self) -> None:
        with self._registry_lock:
            handles = list(self._resident.values())
            self._resident.clear()
        for handle in handles:
            self._release(handle)

    def warmup(self, keys: list[str] | None = None) -> dict[str, float]:
        """Pre-load models so the first real request doesn't pay cold start.

        Called from the FastAPI lifespan hook. Failures are logged and skipped
        rather than raised — a missing SAM checkpoint should not stop the
        service from serving detection.
        """
        targets = keys if keys is not None else self.registered_keys()
        timings: dict[str, float] = {}
        for key in targets:
            try:
                timings[key] = self.handle(key).load_time_ms
            except Exception:  # noqa: BLE001 - warmup is best-effort by design
                logger.warning("warmup failed for %s", key, exc_info=True)
        return timings

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        with self._registry_lock:
            resident = [
                {
                    "key": h.key,
                    "hits": h.hits,
                    "load_time_ms": round(h.load_time_ms, 2),
                    "resident_for_s": round(time.time() - h.loaded_at, 1),
                    "idle_s": round(time.time() - h.last_used_at, 1),
                    "metadata": h.metadata,
                }
                for h in self._resident.values()
            ]
            total_hits = sum(h.hits for h in self._resident.values())
            return {
                "capacity": self._capacity,
                "registered": sorted(self._loaders),
                "resident": resident,
                "cold_loads": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(
                    (total_hits - self._misses) / total_hits, 3
                ) if total_hits else 0.0,
            }

    # --------------------------------------------------------------- internal

    def _evict_if_needed(self) -> None:
        """Caller must hold the registry lock."""
        while len(self._resident) > self._capacity:
            evicted_key, evicted = self._resident.popitem(last=False)
            self._evictions += 1
            logger.info(
                "evicting %s (LRU, idle %.1fs, capacity %d)",
                evicted_key,
                time.time() - evicted.last_used_at,
                self._capacity,
            )
            self._release(evicted)

    @staticmethod
    def _release(handle: ModelHandle[Any]) -> None:
        """Give the adapter a chance to free GPU memory, then drop the ref."""
        close = getattr(handle.model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.warning("close() failed for %s", handle.key, exc_info=True)
        handle.model = None  # type: ignore[assignment]
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
