"""Latency benchmarking across model variants and devices.

The point of this harness is to make the SAM-variant decision with numbers
instead of a preference. MobileSAM versus FastSAM versus full SAM is a real
trade-off (mask quality against ~20x latency) and it changes completely
between CPU and GPU, so the answer has to be measured on the target hardware.

Two things it gets right that a naive timing loop does not:

  * Warmup frames are discarded. The first inference pays lazy CUDA context
    creation, cuDNN autotuning and memory-pool growth; including it inflates
    the mean by an order of magnitude on short runs.
  * p95 is reported alongside the mean. Mean latency hides the frames where
    the detector found eight boxes and the segmenter had to decode eight masks,
    and it is the tail that decides whether real-time playback holds up.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from svqa.vision import frames as frames_mod
from svqa.vision.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchRow:
    stage: str
    model: str
    device: str
    frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    load_time_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def _summarise(
    stage: str,
    model: str,
    device: str,
    timings_ms: list[float],
    load_time_ms: float,
) -> BenchRow:
    mean_ms = statistics.fmean(timings_ms) if timings_ms else 0.0
    return BenchRow(
        stage=stage,
        model=model,
        device=device,
        frames=len(timings_ms),
        mean_ms=round(mean_ms, 2),
        p50_ms=round(_percentile(timings_ms, 50), 2),
        p95_ms=round(_percentile(timings_ms, 95), 2),
        fps=round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
        load_time_ms=round(load_time_ms, 2),
    )


def load_frames(
    video_path: str | Path, count: int, *, resize_width: int | None = 960
) -> list[np.ndarray]:
    """Decode once, reuse across every model, so decode cost is not in the
    measurement and every variant sees identical input."""
    return [
        sampled.image
        for sampled in frames_mod.iter_frames(
            video_path, target_fps=4.0, max_frames=count, resize_width=resize_width
        )
    ]


def benchmark_detectors(
    registry: ModelRegistry,
    keys: list[str],
    images: list[np.ndarray],
    *,
    device: str,
    warmup: int = 4,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for key in keys:
        try:
            handle = registry.handle(key)
        except Exception:  # noqa: BLE001
            logger.warning("skipping detector %s", key, exc_info=True)
            continue
        detector = handle.model

        for image in images[:warmup]:
            detector.detect(image)

        timings: list[float] = []
        for image in images[warmup:] or images:
            started = time.perf_counter()
            detector.detect(image)
            timings.append((time.perf_counter() - started) * 1000.0)

        rows.append(
            _summarise("detect", detector.name, device, timings, handle.load_time_ms)
        )
    return rows


def benchmark_segmenters(
    registry: ModelRegistry,
    keys: list[str],
    images: list[np.ndarray],
    boxes_per_frame: list[list[tuple[float, float, float, float]]],
    *,
    device: str,
    warmup: int = 4,
) -> list[BenchRow]:
    """Segmenters are timed on the *detector's* boxes, not synthetic ones.

    Mask decode cost scales with the number of prompts, so benchmarking with
    one box per frame would understate the real cost of a busy frame.
    """
    rows: list[BenchRow] = []
    for key in keys:
        try:
            handle = registry.handle(key)
        except Exception:  # noqa: BLE001
            logger.warning("skipping segmenter %s", key, exc_info=True)
            continue
        segmenter = handle.model

        for image, boxes in list(zip(images, boxes_per_frame, strict=False))[:warmup]:
            if boxes:
                segmenter.segment_boxes(image, boxes)

        timings: list[float] = []
        for image, boxes in list(zip(images, boxes_per_frame, strict=False))[warmup:]:
            if not boxes:
                continue
            started = time.perf_counter()
            segmenter.segment_boxes(image, boxes)
            timings.append((time.perf_counter() - started) * 1000.0)

        if timings:
            rows.append(
                _summarise(
                    "segment", segmenter.name, device, timings, handle.load_time_ms
                )
            )
    return rows


def run_benchmark(
    registry: ModelRegistry,
    video_path: str | Path,
    *,
    detector_keys: list[str],
    segmenter_keys: list[str],
    device: str,
    frames: int = 32,
    warmup: int = 4,
) -> list[BenchRow]:
    images = load_frames(video_path, frames)
    if not images:
        raise ValueError(f"no frames decoded from {video_path}")
    logger.info("benchmarking on %d frames, device=%s", len(images), device)

    rows = benchmark_detectors(
        registry, detector_keys, images, device=device, warmup=warmup
    )

    # Reuse the first detector's boxes as segmentation prompts.
    boxes_per_frame: list[list[tuple[float, float, float, float]]] = []
    if detector_keys:
        detector = registry.get(detector_keys[0])
        for image in images:
            boxes_per_frame.append([d.bbox for d in detector.detect(image)])

    if boxes_per_frame:
        rows.extend(
            benchmark_segmenters(
                registry,
                segmenter_keys,
                images,
                boxes_per_frame,
                device=device,
                warmup=warmup,
            )
        )
    return rows


def format_table(rows: list[BenchRow]) -> str:
    """Markdown table, for pasting straight into the README."""
    if not rows:
        return "(no results)"
    header = (
        "| stage | model | device | frames | mean ms | p50 | p95 | fps | load ms |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        f"| {r.stage} | {r.model} | {r.device} | {r.frames} | {r.mean_ms} | "
        f"{r.p50_ms} | {r.p95_ms} | {r.fps} | {r.load_time_ms} |"
        for r in rows
    ]
    return "\n".join([header, *lines])
