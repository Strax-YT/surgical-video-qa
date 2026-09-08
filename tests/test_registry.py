"""Registry behaviour: laziness, LRU eviction, concurrency, failure isolation."""

from __future__ import annotations

import threading
import time

import pytest

from svqa.vision.registry import ModelNotRegisteredError, ModelRegistry


class Tracker:
    def __init__(self) -> None:
        self.loads = 0
        self.closes = 0

    def loader(self, name: str, delay: float = 0.0):
        def _load():
            self.loads += 1
            if delay:
                time.sleep(delay)
            return _Model(name, self)
        return _load


class _Model:
    def __init__(self, name: str, tracker: Tracker) -> None:
        self.name = name
        self._tracker = tracker

    def close(self) -> None:
        self._tracker.closes += 1


def test_registration_does_not_load():
    tracker = Tracker()
    registry = ModelRegistry(capacity=2)
    registry.register("a", tracker.loader("a"))
    assert tracker.loads == 0
    assert registry.resident_keys() == []


def test_first_get_loads_then_caches():
    tracker = Tracker()
    registry = ModelRegistry(capacity=2)
    registry.register("a", tracker.loader("a"))
    first = registry.get("a")
    second = registry.get("a")
    assert first is second
    assert tracker.loads == 1


def test_lru_evicts_least_recently_used():
    tracker = Tracker()
    registry = ModelRegistry(capacity=2)
    for key in ("a", "b", "c"):
        registry.register(key, tracker.loader(key))

    registry.get("a")
    registry.get("b")
    registry.get("a")   # 'a' is now more recent than 'b'
    registry.get("c")   # evicts 'b', not 'a'

    assert registry.resident_keys() == ["a", "c"]
    assert tracker.closes == 1
    assert registry.stats()["evictions"] == 1


def test_unknown_key_raises():
    registry = ModelRegistry(capacity=1)
    with pytest.raises(ModelNotRegisteredError):
        registry.get("nope")


def test_failed_load_is_not_cached():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise FileNotFoundError("weights missing")

    registry = ModelRegistry(capacity=2)
    registry.register("broken", broken)

    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            registry.get("broken")

    # Retried, not served from a poisoned cache entry.
    assert calls["n"] == 2
    assert registry.resident_keys() == []


def test_concurrent_get_loads_once():
    """Ten threads racing for the same cold key must trigger exactly one load."""
    tracker = Tracker()
    registry = ModelRegistry(capacity=4)
    registry.register("slow", tracker.loader("slow", delay=0.05))

    results = []
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()
        results.append(registry.get("slow"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tracker.loads == 1
    assert len({id(r) for r in results}) == 1


def test_different_keys_load_in_parallel():
    """A slow load on one key must not block a different key."""
    tracker = Tracker()
    registry = ModelRegistry(capacity=4)
    registry.register("slow", tracker.loader("slow", delay=0.30))
    registry.register("fast", tracker.loader("fast"))

    started = threading.Event()

    def load_slow():
        started.set()
        registry.get("slow")

    thread = threading.Thread(target=load_slow)
    thread.start()
    started.wait()
    time.sleep(0.02)

    began = time.perf_counter()
    registry.get("fast")
    elapsed = time.perf_counter() - began
    thread.join()

    assert elapsed < 0.20, f"fast load waited {elapsed:.3f}s behind the slow one"


def test_warmup_survives_a_broken_loader():
    tracker = Tracker()
    registry = ModelRegistry(capacity=3)
    registry.register("good", tracker.loader("good"))
    registry.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    timings = registry.warmup()

    assert "good" in timings
    assert "bad" not in timings
    assert registry.is_resident("good")
